// A library that puts lines in the layer's VF_SUMMARY file (tests/proxy/summary_fill.py), through the
// layer's vf_summary_add and vf_summary_remove, looked up on the layer's handle as a library would. Its
// functions that give the lines (fills), by kind:
//   count    "count N", N set by fill_set_count
//   bad      its whole buffer: control bytes, bytes that are not UTF-8, a line separator and a bidirectional
//            override, valid UTF-8 between them, "END" in its last bytes, and a return above its size
//   slow     sleeps 300 ms, then "slow"; the times of its calls kept
//   reenter  calls vf_summary_add and vf_summary_remove from inside, says what they returned
//   empty    nothing
//   hang, forks, exits  once armed (fill_arm): never returns; forks (its child ends at once); calls exit(3)
//   segv     once armed, on a thread other than the main one (the layer's): writes through a null pointer
//   slowexit (fill_add_page) once armed, on a thread other than the main one: sleeps 3 s; then gives the number
//            on the page that its ctx points to
//   forkret  once armed, the first time: forks, and its child returns from the fill (fill_child: its pid)
// Each counts its calls. Its destructor, when fill_at_exit asked for it, takes back what it registered
// (as a library removes itself before it goes) and says on standard error how many times its fills
// were called before and after, and what vf_summary_remove returned; with FILL_NO_REMOVE=1 it does not
// take them back (the control: the layer then calls into a library that is gone). When fill_exit_page
// asked for it, it takes back the slowexit fill and the count fill, unmaps the page when the first
// removal returned 0 (as a library frees what its fill reads once its removal says that nothing calls
// it), gives the slow fill 2 s more (a driver's teardown at exit can take as long), and says what both
// removals returned and how many times the count fill was called after them.
#define _GNU_SOURCE
#include <dlfcn.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <signal.h>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <sys/wait.h>
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

static atomic_int removed, after_removal; /* the slowexit case's destructor ran its removals; calls after them */
static size_t fill_count(char *buf, size_t cap, void *ctx)
{
    (void)ctx;
    atomic_fetch_add(&calls, 1);
    if (atomic_load(&removed)) atomic_fetch_add(&after_removal, 1);
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

/* Fills that break the contract, once armed (fill_arm): one that never returns, one that forks (its child ends at
 * once, waited for), one that calls exit(3). in_fill says that the armed one was entered. */
static atomic_int armed, in_fill, forks;
static size_t fill_hang(char *buf, size_t cap, void *ctx)
{
    (void)ctx;
    atomic_fetch_add(&calls, 1);
    if (atomic_load(&armed)) {
        atomic_store(&in_fill, 1);
        for (;;) pause();
    }
    int n = snprintf(buf, cap, "hang not armed\n");
    return n < 0 ? 0 : (size_t)n;
}
static size_t fill_forks(char *buf, size_t cap, void *ctx)
{
    (void)ctx;
    atomic_fetch_add(&calls, 1);
    if (atomic_load(&armed)) {
        atomic_store(&in_fill, 1);
        const pid_t pid = fork();
        if (pid == 0) _exit(0);
        if (pid > 0 && waitpid(pid, NULL, 0) == pid) atomic_fetch_add(&forks, 1);
    }
    int n = snprintf(buf, cap, "forked %d\n", atomic_load(&forks));
    return n < 0 ? 0 : (size_t)n;
}
static size_t fill_exits(char *buf, size_t cap, void *ctx)
{
    (void)ctx;
    atomic_fetch_add(&calls, 1);
    if (atomic_load(&armed)) {
        atomic_store(&in_fill, 1);
        exit(3);
    }
    int n = snprintf(buf, cap, "exits not armed\n");
    return n < 0 ? 0 : (size_t)n;
}
static int is_main(void) { return syscall(SYS_gettid) == getpid(); }
static size_t fill_segv(char *buf, size_t cap, void *ctx)
{
    (void)ctx;
    atomic_fetch_add(&calls, 1);
    if (atomic_load(&armed) && !is_main()) {
        atomic_store(&in_fill, 1);
        *(volatile int *)NULL = 1;
    }
    int n = snprintf(buf, cap, "segv not armed\n");
    return n < 0 ? 0 : (size_t)n;
}
static long *page;
static size_t fill_slowexit(char *buf, size_t cap, void *ctx)
{
    atomic_fetch_add(&calls, 1);
    if (atomic_load(&armed) && !is_main()) {
        atomic_store(&in_fill, 1);
        struct timespec t = {3, 0};
        nanosleep(&t, NULL);
    }
    int n = snprintf(buf, cap, "slowexit %ld\n", *(volatile long *)ctx);
    return n < 0 ? 0 : (size_t)n;
}
static atomic_int child_pid;
static size_t fill_forkret(char *buf, size_t cap, void *ctx)
{
    (void)ctx;
    atomic_fetch_add(&calls, 1);
    if (atomic_load(&armed) && !atomic_exchange(&in_fill, 1)) {
        const pid_t pid = fork();
        if (pid > 0) atomic_store(&child_pid, pid);
    }
    int n = snprintf(buf, cap, "forkret in %d\n", (int)getpid());
    return n < 0 ? 0 : (size_t)n;
}
int fill_child(void) { return atomic_load(&child_pid); }
void fill_arm(int on) { atomic_store(&armed, on); }
int fill_entered(void) { return atomic_load(&in_fill); }

static fill_fn kind(const char *k)
{
    if (!strcmp(k, "count")) return fill_count;
    if (!strcmp(k, "bad")) return fill_bad;
    if (!strcmp(k, "slow")) return fill_slow;
    if (!strcmp(k, "reenter")) return fill_reenter;
    if (!strcmp(k, "empty")) return fill_empty;
    if (!strcmp(k, "hang")) return fill_hang;
    if (!strcmp(k, "forks")) return fill_forks;
    if (!strcmp(k, "exits")) return fill_exits;
    if (!strcmp(k, "segv")) return fill_segv;
    if (!strcmp(k, "forkret")) return fill_forkret;
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
/* The slowexit fill, under NAME, its ctx a page of its own that holds 42. */
int fill_add_page(const char *name)
{
    page = mmap(NULL, 4096, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (page == MAP_FAILED) return -1;
    *page = 42;
    return add(1, name, fill_slowexit, page);
}
static int exit_page;
void fill_exit_page(int on) { exit_page = on; }
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
    if (exit_page) {
        const int r1 = rm(fill_slowexit, page), r2 = rm(fill_count, NULL);
        if (r1 == 0) munmap(page, 4096);
        atomic_store(&removed, 1);
        struct timespec t = {2, 0};
        nanosleep(&t, NULL);
        fprintf(stderr, "fill library: at its end: the slow fill's removal returned %d, the count fill's %d; the count fill "
                "called %d times after them\n", r1, r2, atomic_load(&after_removal));
        return;
    }
    if (!exit_fill) return;
    const char *no = getenv("FILL_NO_REMOVE");
    if (no && !strcmp(no, "1")) return;
    const unsigned long before = atomic_load(&calls);
    const int r = rm(exit_fill, (void *)exit_ctx);
    fprintf(stderr, "fill library: removed at its end: calls before %lu, after %lu, vf_summary_remove returned %d\n",
            before, (unsigned long)atomic_load(&calls), r);
}
