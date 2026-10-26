// SPDX-License-Identifier: GPL-2.0

//! Bedrock - A Rust-based x86-64 hypervisor Linux kernel module

use core::pin::Pin;
use core::sync::atomic::{AtomicBool, Ordering};

use core::mem::size_of;

use kernel::alloc::flags::GFP_KERNEL;
use kernel::bindings;
use kernel::c_str;
use kernel::fs::File;
use kernel::ioctl::_IOW;
use kernel::miscdevice::{MiscDevice, MiscDeviceOptions, MiscDeviceRegistration};
use kernel::prelude::*;

// Internal modules - log must be first for macro availability
#[macro_use]
mod log;
mod c_helpers;
mod ept;
mod factory;
mod instruction_counter;
mod machine;
mod memory;
mod page;
mod vm_file;
mod vmcs;
mod vmx;
mod vmx_asm;
mod vmxon;

use c_helpers::bedrock_copy_from_user;
use factory::{create_vm, KernelFrameAllocator};
use instruction_counter::LinuxInstructionCounter;
use machine::MACHINE;
use vm_file::{create_forked_vm_fd, create_vm_fd, ParentVmArc};
use vmx::traits::{Kernel, Machine, Vmx, VmxCpu};
use vmx::BedrockHandler;
use vmx::VmxoffError;
use vmx::{ForkableVm, ForkedVm, ParentVm};
use vmx_asm::VmxContextExt;
use vmxon::RealVmx;

/// Ioctl magic number for bedrock ('B' for Bedrock).
const BEDROCK_IOC_MAGIC: u32 = b'B' as u32;

/// Configuration for the CREATE_ROOT_VM ioctl.
#[repr(C)]
struct BedrockCreateVmConfig {
    /// Size of guest memory to allocate in bytes.
    memory_size: u64,
    /// TSC frequency in Hz for deterministic time emulation.
    tsc_frequency: u64,
}

/// CREATE_ROOT_VM: takes a `BedrockCreateVmConfig` pointer, returns the VM FD.
const BEDROCK_CREATE_ROOT_VM: u32 = _IOW::<BedrockCreateVmConfig>(BEDROCK_IOC_MAGIC, 0);

/// CREATE_FORKED_VM: takes the parent VM ID, returns the forked VM FD.
const BEDROCK_CREATE_FORKED_VM: u32 = _IOW::<u64>(BEDROCK_IOC_MAGIC, 1);

module! {
    type: Bedrock,
    name: "bedrock",
    authors: ["bedrock-rs"],
    description: "A Rust-based x86-64 hypervisor",
    license: "GPL",
}

/// Register a misc device with a custom mode (the standard
/// `MiscDeviceRegistration::register` can't set it).
fn register_miscdev_with_mode(
    name: &'static kernel::str::CStr,
    mode: u16,
) -> impl PinInit<MiscDeviceRegistration<BedrockFile>, Error> {
    // SAFETY: We properly initialize and register the miscdevice, and the
    // MiscDeviceRegistration's Drop will call misc_deregister.
    unsafe {
        ::pin_init::pin_init_from_closure(move |slot: *mut MiscDeviceRegistration<BedrockFile>| {
            let inner_ptr = slot.cast::<bindings::miscdevice>();

            let opts = MiscDeviceOptions { name };
            inner_ptr.write(opts.into_raw::<BedrockFile>());

            (*inner_ptr).mode = mode;

            kernel::error::to_result(bindings::misc_register(inner_ptr))
        })
    }
}

/// Maximum number of VMs (root + all live forks/checkpoints) tracked at once;
/// a fork past this fails with `ENOSPC`. Cost is heap memory only, so tune to
/// host RAM.
const MAX_TRACKED_VMS: usize = 1024;

// SAFETY: Initialized in module init before first use.
kernel::sync::global_lock! {
    unsafe(uninit) static HANDLER: Mutex<Option<BedrockHandler<'static, RealVmx, MAX_TRACKED_VMS>>> = None;
}

/// Per-open-file private data; VM management lives in the global `HANDLER`.
#[pin_data]
struct BedrockFile {}

/// Handle CREATE_ROOT_VM ioctl - separated to isolate stack usage.
#[inline(never)]
fn handle_create_root_vm(arg: usize) -> Result<isize> {
    let mut config = core::mem::MaybeUninit::<BedrockCreateVmConfig>::uninit();
    // SAFETY: `config` is valid writable memory for a BedrockCreateVmConfig; `arg`
    // is a user pointer and bedrock_copy_from_user performs a bounded copy.
    let not_copied = unsafe {
        bedrock_copy_from_user(
            config.as_mut_ptr().cast::<core::ffi::c_void>(),
            arg as *const core::ffi::c_void,
            size_of::<BedrockCreateVmConfig>() as core::ffi::c_ulong,
        )
    };
    if not_copied != 0 {
        return Err(EFAULT);
    }
    // SAFETY: bedrock_copy_from_user returned 0, so `config` is fully initialized.
    let config = unsafe { config.assume_init() };

    let memory_size = config.memory_size as usize;
    let tsc_frequency = config.tsc_frequency;

    if memory_size == 0 {
        log_err!("Invalid memory size: 0\n");
        return Err(EINVAL);
    }
    if tsc_frequency == 0 {
        log_err!("Invalid TSC frequency: 0\n");
        return Err(EINVAL);
    }

    let vm_id = {
        let mut guard = HANDLER.lock();
        let handler = guard.as_mut().ok_or(ENODEV)?;
        handler.alloc_vm_id().ok_or(ENOSPC)?
    };

    let vm = create_vm(&MACHINE, memory_size, tsc_frequency).ok_or_else(|| {
        log_err!(
            "Failed to create VM {} with {} bytes of memory\n",
            vm_id,
            memory_size
        );
        ENOMEM
    })?;

    let fd = create_vm_fd(vm, vm_id).inspect_err(|e| {
        log_err!("Failed to create VM FD: {:?}\n", e);
    })?;

    log_info!(
        "Created VM {} with fd {} ({} bytes memory)\n",
        vm_id,
        fd,
        memory_size
    );
    Ok(fd as isize)
}

/// Handle CREATE_FORKED_VM ioctl - separated to isolate stack usage.
///
/// The handler lock is held only to allocate the ID, find the parent, and
/// increment its children_count; the expensive fork work (EPT/VMCS cloning)
/// runs outside the lock so forks from the same parent proceed in parallel.
///
/// # Safety Invariants
///
/// - Once children_count > 0, the parent cannot be run (can_run() returns false)
/// - Concurrent forks only READ parent state
/// - The retained parent Arc keeps parent memory alive if its FD closes
#[inline(never)]
fn handle_create_forked_vm(parent_vm_id: u64) -> Result<isize> {
    log_info!("FORK: Starting fork from parent {}\n", parent_vm_id);

    // Phase 1 (locked): allocate ID, find parent, clone parent Arc.
    let (vm_id, parent) = {
        let mut guard = HANDLER.lock();
        let handler = guard.as_mut().ok_or(ENODEV)?;

        let vm_id = handler.alloc_vm_id().ok_or(ENOSPC)?;

        let parent_ref = handler.find_vm_by_id(parent_vm_id).ok_or_else(|| {
            log_err!("Parent VM {} not found\n", parent_vm_id);
            ENOENT
        })?;

        let parent_type = parent_ref.file_type();

        // Increment children_count BEFORE releasing the handler lock.
        let parent = parent_ref;
        match &parent {
            ParentVmArc::Root(parent_file) => {
                parent_file.vm.add_child();
            }
            ParentVmArc::Forked(parent_file) => {
                parent_file.vm.add_child();
            }
        }

        log_info!(
            "FORK: VM {} - found parent {} (type {:?}), incremented children_count\n",
            vm_id,
            parent_vm_id,
            parent_type as u8
        );

        (vm_id, parent)
    }; // Lock released here - expensive work can now proceed in parallel

    // Phase 2 (unlocked): expensive fork work, may run concurrently per parent.
    let fork_result = {
        let mut allocator = KernelFrameAllocator::new(MACHINE.kernel());
        let exit_handler_rip = vmx::VmxContext::exit_handler_addr();
        let instruction_counter = LinuxInstructionCounter::new();

        match &parent {
            ParentVmArc::Root(parent_file) => {
                // children_count was already incremented in phase 1.
                ForkedVm::new_with_incremented_parent(
                    parent_file.as_ref(),
                    &MACHINE,
                    &mut allocator,
                    exit_handler_rip,
                    instruction_counter,
                )
            }
            ParentVmArc::Forked(parent_file) => ForkedVm::new_with_incremented_parent(
                parent_file.as_ref(),
                &MACHINE,
                &mut allocator,
                exit_handler_rip,
                instruction_counter,
            ),
        }
    };

    let forked_vm = match fork_result {
        Ok(vm) => vm,
        Err(e) => {
            log_err!(
                "Failed to create forked VM from parent {}: {:?}\n",
                parent_vm_id,
                e
            );
            // ForkedVm::drop would normally decrement children_count.
            match &parent {
                ParentVmArc::Root(parent_file) => {
                    ParentVm::remove_child(parent_file.as_ref());
                }
                ParentVmArc::Forked(parent_file) => {
                    ParentVm::remove_child(parent_file.as_ref());
                }
            }
            return Err(ENOMEM);
        }
    };

    // Phase 3: create FD (re-acquires lock briefly to register VM).
    log_info!("FORK: Creating FD for forked VM {}\n", vm_id);
    let fd = create_forked_vm_fd(forked_vm, parent, vm_id).inspect_err(|e| {
        log_err!("Failed to create forked VM FD: {:?}\n", e);
    })?;

    log_info!(
        "Created forked VM {} (from parent {}) with fd {}\n",
        vm_id,
        parent_vm_id,
        fd
    );
    Ok(fd as isize)
}

#[vtable]
impl MiscDevice for BedrockFile {
    type Ptr = Pin<KBox<Self>>;

    fn open(_file: &File, _misc: &MiscDeviceRegistration<Self>) -> Result<Pin<KBox<Self>>> {
        log_info!("Bedrock device opened\n");
        KBox::try_pin_init(try_pin_init!(BedrockFile {}), GFP_KERNEL)
    }

    fn ioctl(_me: Pin<&BedrockFile>, _file: &File, cmd: u32, arg: usize) -> Result<isize> {
        match cmd {
            BEDROCK_CREATE_ROOT_VM => handle_create_root_vm(arg),
            BEDROCK_CREATE_FORKED_VM => handle_create_forked_vm(arg as u64),
            _ => {
                log_err!("Unknown ioctl command: {:#x}\n", cmd);
                Err(ENOTTY)
            }
        }
    }
}

/// Whether `bedrock_pmi_nmi_register` succeeded, so drop unregisters it.
static PMI_NMI_REGISTERED: AtomicBool = AtomicBool::new(false);

/// The bedrock kernel module.
#[pin_data(PinnedDrop)]
struct Bedrock {
    #[pin]
    _miscdev: MiscDeviceRegistration<BedrockFile>,
}

impl kernel::InPlaceModule for Bedrock {
    fn init(_module: &'static ThisModule) -> impl PinInit<Self, Error> {
        // SAFETY: The closure initializes all fields and handles errors properly.
        unsafe {
            ::pin_init::pin_init_from_closure(|slot: *mut Self| {
                log_info!("Bedrock module loading...\n");

                // SAFETY: Called exactly once during module initialization.
                HANDLER.init();

                let handler = match BedrockHandler::<RealVmx, MAX_TRACKED_VMS>::new(&MACHINE) {
                    Ok(h) => {
                        log_info!("VMX initialized successfully\n");
                        h
                    }
                    Err(e) => {
                        log_err!("Failed to initialize VMX: {:?}\n", e);
                        return Err(EINVAL);
                    }
                };

                {
                    let mut guard = HANDLER.lock();
                    *guard = Some(handler);
                }

                let miscdev_slot = core::ptr::addr_of_mut!((*slot)._miscdev);
                register_miscdev_with_mode(c_str!("bedrock"), 0o666).__pinned_init(miscdev_slot)?;

                // Last, so no later init failure can unload the module with
                // the handler still registered. PDIR works without it, just
                // less robustly, so a failure is not fatal.
                // SAFETY: Called once from module init; unregistered in drop.
                let ret = c_helpers::bedrock_pmi_nmi_register();
                if ret == 0 {
                    PMI_NMI_REGISTERED.store(true, Ordering::Release);
                } else {
                    log_warn!("Failed to register PMI NMI handler: {}\n", ret);
                }

                log_info!("Bedrock module loaded\n");

                Ok(())
            })
        }
    }
}

#[pinned_drop]
impl PinnedDrop for Bedrock {
    fn drop(self: Pin<&mut Self>) {
        log_info!("Bedrock module unloading...\n");

        if PMI_NMI_REGISTERED.swap(false, Ordering::AcqRel) {
            // SAFETY: Registered in init; unregister waits for running
            // handlers (RCU) before the module text goes away.
            unsafe { c_helpers::bedrock_pmi_nmi_unregister() };
        }

        {
            let mut guard = HANDLER.lock();
            *guard = None;
        }

        match MACHINE.kernel().call_on_all_cpus_with_data(
            &MACHINE,
            |machine| -> Result<(), VmxoffError> {
                let vcpu = RealVmx::current_vcpu();
                vcpu.deinitialize(machine)?;
                Ok(())
            },
        ) {
            Ok(()) => log_info!("VMX deinitialized successfully\n"),
            Err(e) => log_err!("Error during VMX deinit: {:?}\n", e),
        }

        log_info!("Bedrock module unloaded successfully\n");
    }
}
