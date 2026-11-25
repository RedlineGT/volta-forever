// A stall put in front of the C library with LD_PRELOAD by tests/proxy/summary_fill.py: on the one thread armed
// (stall_arm(TID, WHAT)), the next calloc of 4096 bytes (WHAT 1: the buffer the layer gives a fill) or the next
// getpid (WHAT 2: the layer's look at whether it runs in a child) first sleeps 3 s, as a thread that the system
// preempts or that faults there would, then goes on; once. stall_sprung: 1 while it sleeps, 2 after.
#define _GNU_SOURCE
#include <stdatomic.h>
#include <stddef.h>
#include <sys/syscall.h>
#include <sys/types.h>
#include <time.h>
#include <unistd.h>

extern void *__libc_calloc(size_t, size_t);

static atomic_long armed_tid;
static atomic_int armed_what, sprung;

void stall_arm(long tid, int what)
{
    atomic_store(&armed_what, what);
    atomic_store(&armed_tid, tid);
}
int stall_sprung(void) { return atomic_load(&sprung); }

static void stall(int what)
{
    const long tid = atomic_load(&armed_tid);
    if (!tid || atomic_load(&armed_what) != what || syscall(SYS_gettid) != tid) return;
    atomic_store(&armed_tid, 0);
    atomic_store(&sprung, 1);
    struct timespec t = {3, 0};
    nanosleep(&t, NULL);
    atomic_store(&sprung, 2);
}

void *calloc(size_t n, size_t size)
{
    if (n * size == 4096) stall(1);
    return __libc_calloc(n, size);
}

pid_t getpid(void)
{
    stall(2);
    return (pid_t)syscall(SYS_getpid);
}
