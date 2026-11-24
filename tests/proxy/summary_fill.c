// A library that puts lines in the layer's VF_SUMMARY file (tests/proxy/summary_fill.py), through the
// layer's vf_summary_add and vf_summary_remove, looked up on the layer's handle as a library would. Its
// functions that give the lines (fills), by kind:
//   count    "count N", N set by fill_set_count
//   bad      its whole buffer: control bytes, bytes that are not UTF-8, a line separator and a bidirectional
//            override, valid UTF-8 between them, "END" in its last bytes, and a return above its size
//   slow     sleeps 300 ms, then "slow"; the times of its calls kept
//   reenter  calls vf_summary_add and vf_summary_remove from inside, says what they returned
//   empty    nothing
// Each counts its calls. Its destructor, when fill_at_exit asked for it, takes back what it registered
// (as a library removes itself before it goes) and says on standard error how many times its fills
// were called before and after, and what vf_summary_remove returned; with FILL_NO_REMOVE=1 it does not
// take them back (the control: the layer then calls into a library that is gone).
#define _GNU_SOURCE
#include <dlfcn.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <signal.h>
#include <sys/syscall.h>
#include <time.h>
#include <unistd.h>

typedef size_t (*fill_fn)(char *, size_t, void *);
static int (*add)(unsigned, const char *, fill_fn, void *);
static int (*rm)(fill_fn, void *);

static atomic_ulong count, calls;
#define SLOW_CALLS 64
static atomic_llong slow_start[SLOW_CALLS], slow_end[SLOW_CALLS]; /* the times of the slow fill's calls */
static atomic_int slow_calls;
static int reenter_add = -1, reenter_remove = -1;

static long long now_ns(void)
{
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec * 1000000000LL + t.tv_nsec;
}

static size_t fill_count(char *buf, size_t cap, void *ctx)
{
    (void)ctx;
    atomic_fetch_add(&calls, 1);
    int n = snprintf(buf, cap, "count %lu\n", (unsigned long)atomic_load(&count));
    return n < 0 ? 0 : (size_t)n < cap ? (size_t)n : cap;
}

static size_t fill_bad(char *buf, size_t cap, void *ctx)
{
    atomic_fetch_add(&calls, 1);
    /* Control bytes, bytes that are not UTF-8 (a lone 0xff, an unfinished sequence, a surrogate, an overlong
     * encoding), C1 controls, a line separator before text that looks like the layer's (U+2028, at which
     * Python's splitlines cuts) and a right-to-left override (U+202E), with valid UTF-8 between them. */
    static const char piece[] = "ok\x01\x02\t\r\x1b[1m\x7f caf\xc3\xa9 \xff\xfe\xc3 \xe6\xbc\xa2 \xed\xa0\x80\xc0\xaf\xc2\x9b"
                                "\xe2\x80\xa8[volta-forever] forged\xe2\x80\xa9 \xe2\x80\xae" "esrever\n\n\x00 x";
    for (size_t i = 0; i < cap; ++i) buf[i] = piece[i % (sizeof piece - 1)];
    memcpy(buf + cap - 4, "\nEND", 4);
    return ctx ? (size_t)-1 : cap + 1;
}

static size_t fill_slow(char *buf, size_t cap, void *ctx)
{
    (void)ctx;
    atomic_fetch_add(&calls, 1);
    const int i = atomic_fetch_add(&slow_calls, 1) % SLOW_CALLS;
    atomic_store(&slow_start[i], now_ns());
    struct timespec t = {0, 300 * 1000 * 1000};
    nanosleep(&t, NULL);
    int n = snprintf(buf, cap, "slow\n");
    atomic_store(&slow_end[i], now_ns());
    return n < 0 ? 0 : (size_t)n;
}

static size_t fill_reenter(char *buf, size_t cap, void *ctx)
{
    atomic_fetch_add(&calls, 1);
    reenter_add = add(1, "other", fill_reenter, (char *)ctx + 1);
    reenter_remove = rm(fill_reenter, ctx);
    int n = snprintf(buf, cap, "reenter add %d remove %d\n", reenter_add, reenter_remove);
    return n < 0 ? 0 : (size_t)n;
}

static size_t fill_empty(char *buf, size_t cap, void *ctx)
{
    (void)buf, (void)cap, (void)ctx;
    atomic_fetch_add(&calls, 1);
    return 0;
}

static fill_fn kind(const char *k)
{
    if (!strcmp(k, "count")) return fill_count;
    if (!strcmp(k, "bad")) return fill_bad;
    if (!strcmp(k, "slow")) return fill_slow;
    if (!strcmp(k, "reenter")) return fill_reenter;
    if (!strcmp(k, "empty")) return fill_empty;
    return NULL;
}

/* The layer's functions, from the handle of the libcuda.so.1 at LAYER (already loaded). 1 when both are
 * there. */
int fill_bind(const char *layer)
{
    void *h = dlopen(layer, RTLD_NOW | RTLD_NOLOAD);
    if (!h) return 0;
    *(void **)&add = dlsym(h, "vf_summary_add");
    *(void **)&rm = dlsym(h, "vf_summary_remove");
    return add && rm;
}

int fill_add(unsigned abi, const char *name, const char *k, long ctx) { return add(abi, name, kind(k), (void *)ctx); }
int fill_remove(const char *k, long ctx) { return rm(kind(k), (void *)ctx); }
void fill_set_count(unsigned long n) { atomic_store(&count, n); }
unsigned long fill_calls(void) { return atomic_load(&calls); }
long long fill_now(void) { return now_ns(); }
int fill_slow_calls(void) { return atomic_load(&slow_calls); }
long long fill_slow_start(int i) { return atomic_load(&slow_start[i % SLOW_CALLS]); }
long long fill_slow_end(int i) { return atomic_load(&slow_end[i % SLOW_CALLS]); }
int fill_reentered(int which) { return which ? reenter_remove : reenter_add; }

/* A child made without fork()'s handlers (clone(2) as fork does, by the system call itself), while the slow fill
 * holds the layer's lock in the writing thread: the child has that lock held for ever, by a thread it does not have.
 * In it, vf_summary_add and vf_summary_remove must return at once (ECHILD) without taking the lock: the child writes
 * what they returned to fd and ends with _exit. Returns its pid, or -1. */
int fill_raw_child(int fd)
{
    const long pid = syscall(SYS_clone, SIGCHLD, 0L, 0L, 0L, 0L);
    if (pid == 0) {
        int r[2] = {add(1, "child", fill_empty, NULL), rm(fill_count, NULL)};
        if (write(fd, r, sizeof r) != (ssize_t)sizeof r) _exit(2);
        _exit(0);
    }
    return (int)pid;
}

/* Never unloaded from now on, as the bf16 runtime does when it registers. */
int fill_pin(void)
{
    Dl_info info;
    return dladdr((void *)fill_pin, &info) && info.dli_fname &&
           dlopen(info.dli_fname, RTLD_NOW | RTLD_NOLOAD | RTLD_NODELETE) != NULL;
}

/* What the destructor takes back. */
static fill_fn exit_fill;
static long exit_ctx;
void fill_at_exit(const char *k, long ctx)
{
    exit_fill = kind(k);
    exit_ctx = ctx;
}

__attribute__((destructor)) static void fill_fini(void)
{
    if (!exit_fill) return;
    const char *no = getenv("FILL_NO_REMOVE");
    if (no && !strcmp(no, "1")) return;
    const unsigned long before = atomic_load(&calls);
    const int r = rm(exit_fill, (void *)exit_ctx);
    fprintf(stderr, "fill library: removed at its end: calls before %lu, after %lu, vf_summary_remove returned %d\n",
            before, (unsigned long)atomic_load(&calls), r);
}
