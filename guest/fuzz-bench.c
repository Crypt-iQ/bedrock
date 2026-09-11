// SPDX-License-Identifier: GPL-2.0
/*
 * fuzz-bench — a minimal guest harness for the fuzz-input channel.
 *
 * Runs as PID 1 in a throwaway initramfs. It does the least work a snapshot
 * fuzzing harness can do and still be honest:
 *
 *   1. Register a fuzz-input buffer under "fuzzamoto-input".
 *   2. Signal ready (the host checkpoints boot here).
 *   3. Loop: ask for the next testcase, "execute" it (sum its bytes), write the
 *      sum back into the buffer's reserved header word so the host can prove
 *      the guest really saw those bytes, and ask for the next one.
 *
 * The point is to measure what a fork-per-testcase loop costs with everything
 * else stripped out: no target process, no coverage, no I/O. Whatever
 * throughput this reaches is the ceiling for a real harness on this host.
 *
 * Build (see examples/fuzz_fork_bench.rs for the initramfs recipe):
 *   cc -O2 -static -o init guest/fuzz-bench.c
 */

#include <fcntl.h>
#include <sys/mount.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/mman.h>
#include <unistd.h>

#include "libvmcall.h"

/* 1 MB — the hypervisor's per-buffer cap, and far more than any IR program. */
#define INPUT_BUF_SIZE VMCALL_FEEDBACK_BUFFER_MAX_SIZE

/* Header layout, mirroring VMCALL_FUZZ_INPUT_* in libvmcall.h. */
struct fuzz_input_header {
	int64_t result;   /* host writes: input length, or EOF */
	uint64_t status;  /* guest writes: VMCALL_FUZZ_STATUS_* */
	uint64_t aux;     /* guest writes: our checksum (a real harness: msg len) */
	uint64_t reserved;
};

/*
 * Mount /proc.
 *
 * This runs as PID 1 in an initramfs, where nothing is mounted for us. Without
 * it every /proc/cmdline read below fails and each tunable silently defaults to
 * zero — which is exactly the kind of quiet no-op that makes a benchmark report
 * a number for work it never did.
 */
static void mount_proc(void)
{
	(void)mkdir("/proc", 0555);
	if (mount("proc", "/proc", "proc", 0, NULL) != 0)
		vmcall_shutdown();
}

/*
 * Read `ballast=<MB>` out of the kernel command line.
 *
 * Fork cost is a function of how much memory the guest has actually touched,
 * not of the fork call itself, so a 1 MB harness measures a best case no real
 * target will ever see. Ballast lets the bench stand in for a guest with a
 * bitcoind-sized working set.
 */
static long ballast_mb_from_cmdline(void)
{
	char buf[4096];
	int fd = open("/proc/cmdline", O_RDONLY);
	if (fd < 0)
		return 0;
	ssize_t n = read(fd, buf, sizeof(buf) - 1);
	close(fd);
	if (n <= 0)
		return 0;
	buf[n] = '\0';

	const char *p = strstr(buf, "ballast=");
	if (!p)
		return 0;
	long mb = strtol(p + 8, NULL, 10);
	return mb > 0 ? mb : 0;
}

/*
 * Touch `mb` megabytes so the pages are really resident, and keep them mapped
 * for the life of the VM. Every page written here is a page the hypervisor may
 * have to copy on fork.
 */
static void allocate_ballast(long mb)
{
	size_t len = (size_t)mb * 1024 * 1024;
	uint8_t *p = mmap(NULL, len, PROT_READ | PROT_WRITE,
			  MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
	if (p == MAP_FAILED)
		vmcall_shutdown();
	/* One store per page is enough to fault it in and dirty it. */
	for (size_t off = 0; off < len; off += 4096)
		p[off] = (uint8_t)(off >> 12);
}

/*
 * Per-testcase work, so a replay-determinism check has something to check.
 *
 * Summing an input buffer costs one VM exit and touches no state, which makes
 * for a flattering determinism result and a meaningless one. This instead
 * dirties pages all over a scratch region and makes syscalls, driven entirely
 * by the input bytes: the run produces many exits, a guest-memory hash that
 * depends on the input, and timer interrupts landing mid-work. If *that*
 * replays bit-for-bit, the machine is deterministic.
 */
#define WORK_REGION_SIZE (64u * 1024 * 1024)

static uint8_t *g_work_region;

/* Keeps do_work()'s result live without letting it affect the reported sum. */
static volatile uint64_t g_work_sink;

/* Read `work=<iterations>` from the kernel command line. */
static long work_iters_from_cmdline(void)
{
	char buf[4096];
	int fd = open("/proc/cmdline", O_RDONLY);
	if (fd < 0)
		return 0;
	ssize_t n = read(fd, buf, sizeof(buf) - 1);
	close(fd);
	if (n <= 0)
		return 0;
	buf[n] = '\0';

	const char *p = strstr(buf, "work=");
	if (!p)
		return 0;
	long it = strtol(p + 5, NULL, 10);
	return it > 0 ? it : 0;
}

/*
 * Dirty pages and make syscalls, seeded from the input. Returns a checksum of
 * what it did so the compiler cannot elide the work.
 */
static uint64_t do_work(const uint8_t *input, size_t len, long iters)
{
	/* Seed a PRNG from the input so the access pattern follows the testcase. */
	uint64_t state = 0x9e3779b97f4a7c15ULL;
	for (size_t i = 0; i < len; i++)
		state = (state ^ input[i]) * 0x100000001b3ULL;
	state |= 1;

	uint64_t acc = 0;
	for (long i = 0; i < iters; i++) {
		state ^= state << 13;
		state ^= state >> 7;
		state ^= state << 17;

		/*
		 * Scatter a write across the region — dirties a fresh page most
		 * iterations, which is what the CoW page count and the memory
		 * hash are measuring.
		 */
		size_t off = (size_t)(state % WORK_REGION_SIZE);
		g_work_region[off] = (uint8_t)(state >> 32);
		acc += g_work_region[off];

		/*
		 * Periodically leave userspace. getpid() is answered by the
		 * kernel without touching the outside world, so it adds exits
		 * and kernel state transitions without adding nondeterminism.
		 */
		if ((i & 0xff) == 0)
			acc += (uint64_t)getpid();
	}
	return acc;
}

int main(void)
{
	static const char id[] = VMCALL_FUZZ_INPUT_BUFFER_ID;

	mount_proc();

	long work_iters = work_iters_from_cmdline();
	if (work_iters) {
		g_work_region = mmap(NULL, WORK_REGION_SIZE, PROT_READ | PROT_WRITE,
				     MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
		if (g_work_region == MAP_FAILED)
			vmcall_shutdown();
	}

	long ballast = ballast_mb_from_cmdline();
	if (ballast)
		allocate_ballast(ballast);

	/*
	 * The hypervisor translates the buffer by walking the guest page
	 * tables and refuses a non-resident page, so the buffer has to be
	 * faulted in and pinned before registration: MAP_POPULATE faults it,
	 * mlock keeps its GPA stable.
	 */
	void *buf = mmap(NULL, INPUT_BUF_SIZE, PROT_READ | PROT_WRITE,
			 MAP_PRIVATE | MAP_ANONYMOUS | MAP_POPULATE, -1, 0);
	if (buf == MAP_FAILED)
		vmcall_shutdown();
	if (mlock(buf, INPUT_BUF_SIZE) != 0)
		vmcall_shutdown();
	memset(buf, 0, INPUT_BUF_SIZE);

	vmcall_u64 slot = vmcall_register_feedback_buffer(buf, INPUT_BUF_SIZE,
							  id, sizeof(id) - 1);
	/* Every error code is >= VMCALL_ERR - 4; a real slot index is small. */
	if (slot >= VMCALL_ERR - 4ULL)
		vmcall_shutdown();

	/* Boot is done. The host checkpoints here. */
	vmcall_ready();

	volatile struct fuzz_input_header *hdr = buf;
	const uint8_t *data = (const uint8_t *)buf + VMCALL_FUZZ_INPUT_HEADER_LEN;

	for (;;) {
		/*
		 * Ask for work. On a forked VM this is where execution
		 * resumes, with the host's input already in the buffer. It is
		 * also how the host learns the previous testcase finished.
		 */
		vmcall_fuzz_next_input();

		int64_t len = hdr->result;
		if (len == VMCALL_FUZZ_INPUT_EOF)
			break;
		if (len < 0 || (uint64_t)len > INPUT_BUF_SIZE - VMCALL_FUZZ_INPUT_HEADER_LEN)
			break;

		/*
		 * Stand-in for executing the testcase. Touching every byte is
		 * the part a real harness would also do, and it keeps the
		 * compiler from optimising the read away.
		 */
		uint64_t sum = 0;
		for (int64_t i = 0; i < len; i++)
			sum += data[i];

		/*
		 * The host verifies `sum` alone, so the work below cannot mask
		 * a wrong input; it exists to give the machine something real
		 * to be deterministic about.
		 */
		if (work_iters)
			g_work_sink += do_work(data, (size_t)len, work_iters);

		/*
		 * Report the testcase as clean and hand the host proof we saw
		 * this exact input. Both are read after the *next*
		 * vmcall_fuzz_next_input(), which is what tells the host this
		 * execution finished.
		 */
		hdr->aux = sum;
		hdr->status = VMCALL_FUZZ_STATUS_OK;
	}

	vmcall_shutdown();
	for (;;)
		__asm__ volatile("hlt");
}
