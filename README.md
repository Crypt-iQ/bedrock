# Bedrock

An experimental x86-64 hypervisor purpose-built for deterministic software
testing.

Bedrock uses Intel VT-x to run guest VMs with fully emulated time (TSC),
controlled randomness (RDRAND/RDSEED), various other device emulation, and
copy-on-write VM forking - enabling reproducible execution for deterministic
testing.

<a href="https://asciinema.org/a/icy1rkUAHbCEQsRN" target="_blank"><img src="https://asciinema.org/a/icy1rkUAHbCEQsRN.svg" /></a>

## Architecture

```
┌────────────────────────────────────────────────────┐
│                    User Space                      │
│                                                    │
│  bedrock-vm      Rust library for VM control       │
│                                                    │
│                        │ ioctl                     │
├────────────────────────┼───────────────────────────┤
│                        ▼                           │
│                   Kernel Space                     │
│                                                    │
│  bedrock.ko      Kernel module (/dev/bedrock)      │
│                  - VMX setup and VM execution      │
│                  - EPT memory virtualization       │
│                  - Deterministic device emulation  │
│                  - ...                             │
│                                                    │
└────────────────────────────────────────────────────┘
```

## Requirements

- [Linux 6.18] host kernel built with `CONFIG_RUST=y` and rustc >= 1.87
- Patched linux 6.18 guest kernel (see [guest-patches/](guest/patches/))
- Bedrock requires Intel CPUs that support either `EPT-friendly PEBS` or
  the `PDIR` feature.
  
  The following CPUs have been confirmed to work:
  - `Intel(R) Xeon(R) Gold 5412U` (rented on [Hetzner])
  - `Intel(R) Xeon(R) Silver 4310` (rented on [Worldstream])
  - `Intel(R) Core(TM) i7-7700` (rented on [Hetzner])
  - `Intel(R) Core(TM) i7-6700` (rented on [Hetzner])
  - `{r,m}7i.metal-24xl` instances on [AWS].

[Linux 6.18]: https://github.com/torvalds/linux/tree/v6.18
[Hetzner]: https://www.hetzner.com/dedicated-rootserver/
[Worldstream]: https://www.worldstream.com/en/dedicated-servers/
[AWS]: https://aws.amazon.com/de/ec2/instance-types/

## CI

CI uses [RunsOn](https://runs-on.com/) with `m7i` AWS instances.

---

*This project was created with heavy assistance from LLMs. Might freeze/hang or
otherwise corrupt host machine, run at your own risk.*
