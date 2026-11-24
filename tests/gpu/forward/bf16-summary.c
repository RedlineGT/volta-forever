/* The bf16 runtime's counts in the layer's VF_SUMMARY file (crates/vf-fwdgen/bf16tc/runtime.c, summary_fill; the
 * layer's vf_summary_add, crates/vf-proxy/src/summary.rs). This program loads nothing of CUDA itself: it makes a
 * child process, which opens the layer's libcuda.so.1 and the variant libcublas.so.13 with dlopen and makes bf16
 * calls through the variant (three served by the tensor-core path, two by the skinny path, one forwarded), with
 * VF_SUMMARY=FILE and VF_SUMMARY_INTERVAL=1, and checks FILE.PID of that child from outside. The case, from
 * VF_BF16_SUMMARY_CASE (tests/gpu/forward-bf16.sh runs each in a process of its own):
 *   killed         the child sleeps after its calls and is killed with SIGKILL: its file holds the runtime's lines,
 *                  "[libvf_cublas_bf16.so] " in front, as the runtime's fill gives them (its test build's export),
 *                  with the counts of the calls made, and says that the fallback's counters are read at exit only
 *   order-layer    the layer loaded first, the child exits
 *   order-runtime  the runtime loaded first (cublasCreate before anything of the driver; cuBLAS 12.9 loads the layer)
 *                  At exit, which destructor came first is read from the runtime's removal: the layer's (glibc 2.39,
 *                  layer loaded first), whose last write calls the fill (live lines), the removal after it calling
 *                  nothing; or the runtime's (runtime loaded first), whose removal calls the fill once more, and the
 *                  layer's last write holds that last report ("[libvf_cublas_bf16.so, last report] "). Either way
 *                  the file's lines are the exit report's at VF_LOG_LEVEL=2, the line of the fallback's counters
 *                  included (read by the exit report before); tests/gpu/forward-bf16.sh requires both orders
 *   fork           children made by fork() after the calls: vf_summary_add and vf_summary_remove do nothing there
 *                  (ECHILD), nor the runtime's removal when they exit; they end, write no file, and the fill is not
 *                  called in them
 *   dlclose        the variant unloaded by dlclose while the layer's thread writes, then the runtime too as far as
 *                  dlclose can (every reference given back): the runtime stays (RTLD_NODELETE), its lines are still
 *                  written, the process lives on
 * With VF_BF16_SUMMARY_NO_FILE=1 the child runs without VF_SUMMARY (a control: there is no file). The test runtime's
 * controls must make it fail: summary-off (no registration), no-pin (dlclose unloads the runtime; its destructor
 * removes it, and its last report is written), no-pin,no-remove (the layer's thread calls into the unloaded runtime:
 * the child ends by SIGSEGV). Prints "ok bf16-summary CASE". */
#define _GNU_SOURCE
#include <cublas_v2.h>
#include <cuda.h>
#include <dlfcn.h>
#include <errno.h>
#include <fcntl.h>
#include <signal.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/select.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>

static int failures;
#define CHECK(cond, ...)                                 \
    do {                                                 \
        if (!(cond)) {                                   \
            ++failures;                                  \
            printf("FAIL  %s:%d: ", __FILE__, __LINE__); \
            printf(__VA_ARGS__);                         \
            printf("\n");                                \
        }                                                \
    } while (0)

static const char *const RUNTIME = "libvf_cublas_bf16.so";
enum { R_SHAPE = 4, R_N = 26 }; /* the runtime's reasons to forward: the fifth of its 26 (runtime.c's order) */
static char dir[4096], file[4300], err_path[4200];

static double now(void)
{
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (double)t.tv_sec + 1e-9 * (double)t.tv_nsec;
}

static void nap(double s)
{
    struct timespec t = {(time_t)s, (long)((s - (double)(time_t)s) * 1e9)};
    nanosleep(&t, NULL);
}

/* The whole of a file, or NULL. */
static char *slurp(const char *path)
{
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    size_t cap = 1 << 16, n = 0;
    char *b = malloc(cap + 1);
    size_t got;
    while ((got = fread(b + n, 1, cap - n, f)) > 0)
        if ((n += got) == cap) b = realloc(b, (cap *= 2) + 1);
    fclose(f);
    b[n] = 0;
    return b;
}

/* The lines of text that begin with prefix, without it, each ended by a newline (malloc'd). */
static char *lines_of(const char *text, const char *prefix)
{
    size_t cap = strlen(text) + 1, n = 0;
    char *out = malloc(cap);
    out[0] = 0;
    for (const char *p = text; p && *p;) {
        const char *e = strchr(p, '\n');
        const size_t len = e ? (size_t)(e - p) : strlen(p);
        if (!strncmp(p, prefix, strlen(prefix)) && len >= strlen(prefix)) {
            memcpy(out + n, p + strlen(prefix), len - strlen(prefix));
            n += len - strlen(prefix);
            out[n++] = '\n';
            out[n] = 0;
        }
        p = e ? e + 1 : NULL;
    }
    return out;
}

/* ---- the child: CUDA through dlopen ---- */
static struct {
    __typeof__(cuInit) *Init;
    __typeof__(cuDevicePrimaryCtxRetain) *Retain;
    __typeof__(cuCtxSetCurrent) *SetCurrent;
    __typeof__(cuCtxSynchronize) *Synchronize;
    __typeof__(cuMemAlloc_v2) *Alloc;
    __typeof__(cuMemsetD16_v2) *Set16;
    __typeof__(cublasCreate_v2) *Create;
    __typeof__(cublasDestroy_v2) *Destroy;
    __typeof__(cublasGemmEx) *GemmEx;
    int (*add)(unsigned, const char *, size_t (*)(char *, size_t, void *), void *);
    int (*remove)(size_t (*)(char *, size_t, void *), void *);
    int (*counters)(int, unsigned long long *, int);
    size_t (*text)(char *, size_t);
    int (*summary)(unsigned long long *, int);
} f;
static void *layer, *blas;
static int report_fd = -1;

/* To the parent, through the pipe: one line. */
static void tell(const char *fmt, ...)
{
    char b[8192];
    va_list ap;
    va_start(ap, fmt);
    const int n = vsnprintf(b, sizeof b, fmt, ap);
    va_end(ap);
    if (n > 0 && write(report_fd, b, (size_t)n < sizeof b ? (size_t)n : sizeof b - 1) < 0) _exit(3);
}

static void die(const char *what)
{
    tell("error %s: %s\n", what, dlerror() ? dlerror() : "");
    _exit(2);
}

#define SYM(h, field, name)                                       \
    do {                                                          \
        if (!(*(void **)&f.field = dlsym(h, name))) die(name);    \
    } while (0)

static void open_layer(int noload)
{
    layer = dlopen("libcuda.so.1", RTLD_NOW | RTLD_GLOBAL | (noload ? RTLD_NOLOAD : 0));
    if (!layer) die("libcuda.so.1");
    SYM(layer, Init, "cuInit");
    SYM(layer, Retain, "cuDevicePrimaryCtxRetain");
    SYM(layer, SetCurrent, "cuCtxSetCurrent");
    SYM(layer, Synchronize, "cuCtxSynchronize");
    SYM(layer, Alloc, "cuMemAlloc_v2");
    SYM(layer, Set16, "cuMemsetD16_v2");
    *(void **)&f.add = dlsym(layer, "vf_summary_add");
    *(void **)&f.remove = dlsym(layer, "vf_summary_remove");
}

static void current(void)
{
    CUcontext ctx;
    if (f.Init(0) || f.Retain(&ctx, 0) || f.SetCurrent(ctx)) die("the primary context");
}

/* The calls: TN 1536 x 128 x 4096 three times (the tensor-core path), NT 1536 x 8 x 1920 twice (the skinny path),
 * TN 1536 x 128 x 8 once (forwarded: k < 16). */
static void work(int runtime_first)
{
    if (!runtime_first) {
        open_layer(0);
        current();
    }
    blas = dlopen("libcublas.so.13", RTLD_NOW | RTLD_LOCAL);
    if (!blas) die("libcublas.so.13");
    SYM(blas, Create, "cublasCreate_v2");
    SYM(blas, Destroy, "cublasDestroy_v2");
    SYM(blas, GemmEx, "cublasGemmEx");
    cublasHandle_t h;
    if (f.Create(&h)) die("cublasCreate");
    if (runtime_first) { /* the first handle loaded the runtime, then cuBLAS 12.9 the driver: the layer */
        open_layer(1);
        current();
        f.Destroy(h);
        if (f.Create(&h)) die("cublasCreate");
    }
    void *rt = dlopen(RUNTIME, RTLD_NOW | RTLD_NOLOAD);
    if (!rt) die(RUNTIME);
    SYM(rt, counters, "vf_bf16_test_counters");
    /* (A test runtime without the summary, as before it, goes on: its file is checked all the same.) */
    *(void **)&f.text = dlsym(rt, "vf_bf16_test_summary_text");
    *(void **)&f.summary = dlsym(rt, "vf_bf16_test_summary");
    CUdeviceptr a, b, c;
    if (f.Alloc(&a, 4096 * 1536 * 2) || f.Alloc(&b, 4096 * 128 * 2) || f.Alloc(&c, 1536 * 128 * 4)) die("cuMemAlloc");
    if (f.Set16(a, 0x3f80, 4096 * 1536) || f.Set16(b, 0x3f80, 4096 * 128)) die("cuMemsetD16");
    const float one = 1.0f, zero = 0.0f;
    int st = 0;
    for (int i = 0; i < 3; ++i)
        st |= f.GemmEx(h, CUBLAS_OP_T, CUBLAS_OP_N, 1536, 128, 4096, &one, (void *)a, CUDA_R_16BF, 4096, (void *)b,
                       CUDA_R_16BF, 4096, &zero, (void *)c, CUDA_R_32F, 1536, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
    for (int i = 0; i < 2; ++i)
        st |= f.GemmEx(h, CUBLAS_OP_N, CUBLAS_OP_T, 1536, 8, 1920, &one, (void *)a, CUDA_R_16BF, 1536, (void *)b,
                       CUDA_R_16BF, 8, &zero, (void *)c, CUDA_R_32F, 1536, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
    st |= f.GemmEx(h, CUBLAS_OP_T, CUBLAS_OP_N, 1536, 128, 8, &one, (void *)a, CUDA_R_16BF, 8, (void *)b, CUDA_R_16BF,
                   8, &zero, (void *)c, CUDA_R_32F, 1536, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
    if (st || f.Synchronize()) die("the calls");
    f.Destroy(h);
    unsigned long long v[4 + R_N], s[2] = {0, 0};
    memset(v, 0, sizeof v);
    f.counters(0, v, 4 + R_N);
    if (f.summary) f.summary(s, 2);
    char text[4096];
    const size_t n = f.text ? f.text(text, sizeof text) : 0;
    tell("counts %llu %llu %llu %llu %llu registered %llu\n", v[0], v[1], v[2], v[3], v[4 + R_SHAPE], s[0]);
    for (const char *p = text; p < text + n;) { /* the fill's text, one line each */
        const char *e = memchr(p, '\n', (size_t)(text + n - p));
        if (!e) e = text + n;
        tell("fill %.*s\n", (int)(e - p), p);
        p = e + 1;
    }
    tell("done\n");
}

/* Whether the file holds every line of want, each with prefix in front. */
static int file_has(const char *want, const char *prefix)
{
    char *text = slurp(file);
    if (!text) return 0;
    char *mine = lines_of(text, prefix);
    const int ok = !strcmp(mine, want);
    free(mine);
    free(text);
    return ok;
}

/* Until the file holds the runtime's line of cublasGemmEx (20 s at most). */
static void wait_for_lines(void)
{
    const double end = now() + 20;
    while (now() < end) {
        char *t = slurp(file);
        const int seen = t && strstr(t, "[libvf_cublas_bf16.so] cublasGemmEx: ");
        free(t);
        if (seen) return;
        nap(0.1);
    }
}

static void child_dlclose(void)
{
    work(0);
    wait_for_lines();
    if (dlclose(blas)) die("dlclose of the variant");
    void *rt = dlopen(RUNTIME, RTLD_NOW | RTLD_NOLOAD);
    tell("after the variant's dlclose: the runtime %s\n", rt ? "loaded" : "unloaded");
    if (rt) dlclose(rt);
    nap(2.2); /* two writes of the layer's thread */
    /* Every reference given back, as far as dlclose takes them (the variant's, and any other: dlclose of an object
     * that is pinned does nothing), each time with a handle of this moment, never one of an object unloaded since. */
    for (int i = 0; i < 8; ++i) {
        void *h = dlopen(RUNTIME, RTLD_NOW | RTLD_NOLOAD);
        if (!h) break;
        dlclose(h); /* this one */
        dlclose(h); /* another's */
    }
    rt = dlopen(RUNTIME, RTLD_NOW | RTLD_NOLOAD);
    tell("after every reference: the runtime %s\n", rt ? "loaded" : "unloaded");
    nap(2.2);
    tell("alive\n");
}

static size_t nothing(char *buf, size_t cap, void *ctx)
{
    (void)buf, (void)cap, (void)ctx;
    return 0;
}

static void child_fork(void)
{
    pid_t pids[10];
    for (int i = 0; i < 10; ++i) {
        const pid_t pid = fork();
        if (pid == 0) {
            if (!f.add || !f.remove || !f.summary) exit(1);
            const int r1 = f.add(1, "child", nothing, NULL), r2 = f.remove(nothing, NULL);
            unsigned long long s0[2], s1[2];
            f.summary(s0, 2);
            nap(0.3);
            f.summary(s1, 2);
            exit(r1 == ECHILD && r2 == ECHILD && s1[1] == s0[1] ? 0 : 1); /* exit(): the runtime's destructor runs */
        }
        pids[i] = pid;
        nap(0.15);
    }
    double end = now() + 30;
    int ended = 0, ok = 0;
    for (int i = 0; i < 10; ++i) {
        int status = 0;
        while (waitpid(pids[i], &status, WNOHANG) == 0) {
            if (now() > end) {
                kill(pids[i], SIGKILL);
                waitpid(pids[i], &status, 0);
                status = -1;
                break;
            }
            nap(0.05);
        }
        ended += status != -1;
        ok += status != -1 && WIFEXITED(status) && WEXITSTATUS(status) == 0;
    }
    char list[512] = "";
    for (int i = 0; i < 10; ++i) snprintf(list + strlen(list), sizeof list - strlen(list), " %d", (int)pids[i]);
    tell("children ended %d ok %d:%s\n", ended, ok, list);
}

/* ---- the parent ---- */
static char *child_said;
static size_t child_len;

static int start(const char *mode, int *fd, pid_t *pid)
{
    int p[2];
    if (pipe(p)) return 0;
    *pid = fork();
    if (*pid == 0) {
        close(p[0]);
        report_fd = p[1];
        const int e = open(err_path, O_CREAT | O_WRONLY | O_TRUNC, 0600);
        if (e >= 0) dup2(e, 2);
        if (!strcmp(mode, "killed") || !strcmp(mode, "order-layer")) {
            work(0);
            if (!strcmp(mode, "killed")) {
                nap(600);
            }
        } else if (!strcmp(mode, "order-runtime")) {
            work(1);
        } else if (!strcmp(mode, "fork")) {
            work(0);
            wait_for_lines();
            child_fork();
        } else if (!strcmp(mode, "dlclose")) {
            child_dlclose();
        }
        exit(0);
    }
    close(p[1]);
    *fd = p[0];
    return *pid > 0;
}

/* What the child said so far (more read when there is some), and whether it said `what`. */
static int heard(int fd, const char *what, double seconds)
{
    const double end = now() + seconds;
    for (;;) {
        if (child_said && strstr(child_said, what)) return 1;
        fd_set r;
        FD_ZERO(&r);
        FD_SET(fd, &r);
        struct timeval tv = {0, 100000};
        if (select(fd + 1, &r, NULL, NULL, &tv) > 0) {
            char b[4096];
            const ssize_t n = read(fd, b, sizeof b);
            if (n <= 0) return child_said && strstr(child_said, what);
            child_said = realloc(child_said, child_len + (size_t)n + 1);
            memcpy(child_said + child_len, b, (size_t)n);
            child_len += (size_t)n;
            child_said[child_len] = 0;
        }
        if (now() > end) return 0;
    }
}

/* The fill's text that the child reported (its "fill " lines). */
static char *reported_fill(void) { return lines_of(child_said ? child_said : "", "fill "); }

static void check_counts(void)
{
    unsigned long long calls = 0, skinny = 0, tc = 0, failed = 0, shape = 0, reg = 0;
    const char *c = child_said ? strstr(child_said, "counts ") : NULL;
    CHECK(c && sscanf(c, "counts %llu %llu %llu %llu %llu registered %llu", &calls, &skinny, &tc, &failed, &shape,
                      &reg) == 6,
          "the child did not say its counts: %s", child_said ? child_said : "");
    CHECK(calls == 6 && skinny == 2 && tc == 3 && failed == 0 && shape == 1,
          "the calls were not served as made: %llu calls, skinny %llu, tensor-core %llu, failed %llu, shape %llu", calls,
          skinny, tc, failed, shape);
    CHECK(reg == 1 || getenv("VF_BF16_CONTROL"), "the runtime did not register with the layer");
}

int main(void)
{
    const char *mode = getenv("VF_BF16_SUMMARY_CASE");
    if (!mode) mode = "killed";
    const char *tmp = getenv("TMPDIR");
    snprintf(dir, sizeof dir, "%s/vf-bf16-summary.XXXXXX", tmp && *tmp ? tmp : "/tmp");
    if (!mkdtemp(dir)) {
        printf("FAIL  cannot make a directory under %s\n", tmp ? tmp : "/tmp");
        return 1;
    }
    char base[4200];
    snprintf(base, sizeof base, "%s/summary", dir);
    snprintf(err_path, sizeof err_path, "%s/stderr", dir);
    const char *no_file = getenv("VF_BF16_SUMMARY_NO_FILE");
    if (!no_file || strcmp(no_file, "1")) setenv("VF_SUMMARY", base, 1);
    setenv("VF_SUMMARY_INTERVAL", "1", 1);
    setenv("VF_BF16_SUMMARY_SAY", "1", 1);
    if (!strcmp(mode, "fork")) setenv("VF_LOG_LEVEL", "1", 1); /* the children's exit report reads nothing of the GPU */
    int fd;
    pid_t pid;
    if (!start(mode, &fd, &pid)) {
        printf("FAIL  cannot start the child\n");
        return 1;
    }
    snprintf(file, sizeof file, "%s.%d", base, (int)pid);
    const int done = heard(fd, "done\n", 120);
    CHECK(done, "the child did not make its calls: %s", child_said ? child_said : "(nothing said)");
    if (done) check_counts();
    char *fill = reported_fill();
    int status = 0;
    if (!strcmp(mode, "killed")) {
        const double end = now() + 20;
        while (!file_has(fill, "[libvf_cublas_bf16.so] ") && now() < end) nap(0.1);
        nap(1.2); /* past one more write */
        kill(pid, SIGKILL);
        waitpid(pid, &status, 0);
        CHECK(WIFSIGNALED(status) && WTERMSIG(status) == SIGKILL, "the child did not end by SIGKILL (status %d)", status);
        char *text = slurp(file);
        CHECK(text, "no file %s", file);
        if (text) {
            char *mine = lines_of(text, "[libvf_cublas_bf16.so] ");
            CHECK(strstr(text, ", running for ") && !strstr(text, "at its exit"), "the file is not that of a running process:\n%s",
                  text);
            CHECK(!strcmp(mine, fill) && *fill, "the file's lines of the runtime are\n%s\nnot what its fill gives\n%s", mine,
                  fill);
            CHECK(strstr(mine, "cublasGemmEx: 6 calls, 6 with bf16 operands; skinny path 2, tensor-core path 3 (0 in graph "
                               "captures), failed after writing 0; forwarded: shape 1,\n") &&
                      strstr(mine, "tensor-core path: the fallback's counters are on the GPU, read at exit only"),
                  "the file does not hold the counts of the calls made, or what the fallback's are:\n%s", mine);
            printf("the file of the killed child:\n%s", text);
            free(mine);
            free(text);
        }
    } else if (!strcmp(mode, "order-layer") || !strcmp(mode, "order-runtime")) {
        const int runtime_loaded_first = !strcmp(mode, "order-runtime");
        waitpid(pid, &status, 0);
        CHECK(WIFEXITED(status) && WEXITSTATUS(status) == 0, "the child ended with status %d", status);
        char *text = slurp(file), *err = slurp(err_path);
        CHECK(text, "no file %s", file);
        if (text && err) {
            /* Which destructor came first, from the runtime's removal: before the layer's last write it calls the fill
             * once more (and that last report is what the file holds); after it, it calls nothing. */
            unsigned long long before = 0, after = 0;
            int r = -1;
            const char *s = strstr(err, "test build: vf_summary_remove returned ");
            CHECK(s && sscanf(s, "test build: vf_summary_remove returned %d; the summary's fill called %llu times before it, "
                                 "%llu after", &r, &before, &after) == 3 &&
                      r == 0 && (after == before || after == before + 1),
                  "%s: the runtime's removal: %s", mode, s ? s : "(not said)");
            const int runtime_first = after == before + 1;
            const char *live = "[libvf_cublas_bf16.so] ", *kept = "[libvf_cublas_bf16.so, last report] ";
            char *mine = lines_of(text, runtime_first ? kept : live), *other = lines_of(text, runtime_first ? live : kept);
            /* The exit report's lines of the entry points and of the fallback's counters, at VF_LOG_LEVEL=2. */
            char *said = lines_of(err, "[volta-forever] libvf_cublas_bf16.so: "), *report = calloc(1, strlen(said) + 1);
            for (char *p = said; *p;) {
                char *e = strchr(p, '\n');
                if (!strncmp(p, "cublas", 6) || !strncmp(p, "tensor-core path: ", 18))
                    strncat(report, p, (size_t)(e - p + 1));
                p = e + 1;
            }
            CHECK(strstr(text, "at its exit\n") != NULL, "the file is not that of an exit:\n%s", text);
            CHECK(*report && !strcmp(mine, report) && !*other,
                  "%s: the file's lines of the runtime (%s) are\n%s\nnot the exit report's\n%s(other lines: %s)", mode,
                  runtime_first ? "its last report" : "live", mine, report, other);
            /* (With the runtime loaded before the driver, the exit report runs after the driver's own teardown at
             * exit, registered when it was loaded: it says that it could not read them.) */
            CHECK(strstr(report, "tensor-core path: of its 3 calls, ") ||
                      (runtime_loaded_first && strstr(report, "tensor-core path: the fallback's counters could not be read")),
                  "the exit report has no line of the fallback's counters:\n%s", report);
            printf("the file at the exit (%s):\n%s", mode, text);
            printf("the runtime's removal: %.*s\n", s ? (int)strcspn(s, "\n") : 0, s ? s : "");
            printf("at exit, the %s destructor came first\n", runtime_first ? "runtime's" : "layer's");
            free(mine);
            free(other);
            free(said);
            free(report);
        }
        free(text);
        free(err);
    } else if (!strcmp(mode, "fork")) {
        CHECK(heard(fd, "children ", 60), "the child did not say how its children ended: %s", child_said);
        waitpid(pid, &status, 0);
        CHECK(WIFEXITED(status) && WEXITSTATUS(status) == 0, "the child ended with status %d", status);
        const char *c = child_said ? strstr(child_said, "children ") : NULL;
        int ended = 0, ok = 0, n = 0;
        CHECK(c && sscanf(c, "children ended %d ok %d:%n", &ended, &ok, &n) == 2 && ended == 10 && ok == 10,
              "the children made by fork(): %s", c ? c : "(not said)");
        for (const char *p = c ? c + n : ""; *p && *p != '\n';) {
            char *e;
            const long child = strtol(p, &e, 10);
            if (e == p) break;
            char other[4300];
            snprintf(other, sizeof other, "%s.%ld", base, child);
            struct stat sb;
            CHECK(stat(other, &sb) != 0, "a child made by fork() wrote %s", other);
            p = e;
        }
        char *err = slurp(err_path);
        int removals = 0;
        for (const char *p = err; p && (p = strstr(p, "test build: vf_summary_remove returned ")); ++p) {
            int r = -1;
            unsigned long long before = 0, after = 0;
            if (sscanf(p, "test build: vf_summary_remove returned %d; the summary's fill called %llu times before it, %llu "
                          "after", &r, &before, &after) == 3 && r == ECHILD && after == before)
                ++removals;
        }
        CHECK(removals == 10, "%d children's ends removed nothing and called nothing (ECHILD), not 10:\n%s", removals,
              err ? err : "");
        printf("children made by fork(): %d ended, %d with vf_summary_add and vf_summary_remove refused (ECHILD) and "
               "the fill not called; at their exit, %d removals refused with the fill not called\n", ended, ok, removals);
        char *text = slurp(file);
        CHECK(text && strstr(text, "at its exit\n") && strstr(text, "cublasGemmEx: 6 calls"),
              "the file of the process that made them:\n%s", text ? text : "(none)");
        free(err);
        free(text);
    } else if (!strcmp(mode, "dlclose")) {
        CHECK(heard(fd, "alive\n", 60), "the child did not live on: %s", child_said ? child_said : "");
        waitpid(pid, &status, 0);
        CHECK(WIFEXITED(status) && WEXITSTATUS(status) == 0, "the child ended %s %d", WIFSIGNALED(status) ? "by signal" : "with status",
              WIFSIGNALED(status) ? WTERMSIG(status) : WEXITSTATUS(status));
        CHECK(child_said && strstr(child_said, "after the variant's dlclose: the runtime loaded"),
              "the variant's dlclose unloaded the runtime: %s", child_said);
        CHECK(child_said && strstr(child_said, "after every reference: the runtime loaded"),
              "the runtime was unloaded (not pinned): %s", child_said ? child_said : "");
        char *text = slurp(file);
        char *mine = text ? lines_of(text, "[libvf_cublas_bf16.so] ") : NULL;
        CHECK(mine && strstr(mine, "cublasGemmEx: 6 calls") && !strstr(text, "last report"),
              "the runtime's lines at exit:\n%s", text ? text : "(no file)");
        if (text) printf("the file at the exit (dlclose):\n%s", text);
        free(mine);
        free(text);
    }
    free(fill);
    if (failures) {
        char *err = slurp(err_path);
        printf("the child's standard error:\n%s", err ? err : "(none)");
        free(err);
    }
    char cmd[4300];
    snprintf(cmd, sizeof cmd, "rm -rf '%s'", dir);
    if (system(cmd)) printf("note  %s not removed\n", dir);
    if (failures) {
        printf("%d checks failed\n", failures);
        return 1;
    }
    printf("ok bf16-summary %s\n", mode);
    return 0;
}
