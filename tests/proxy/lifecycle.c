// What the layer keeps over the life of a process, and what it does at its end, without a GPU.
//
// A stand-in for the driver (lifecycle_mock_driver.c) gives every module a handle of its own. Each
// case runs in a process of its own, as the layer reads its settings when it is loaded:
//   - fork_exit: a child made by fork() while another thread of the parent waits in a write of
//     the log, and that ends with exit(), ends. (What the layer would report at the child's end
//     is its parent's, and the lock of the log stays held in the child for ever.)
//   - contexts: what the layer keeps of the modules of a context is dropped when the context is
//     destroyed (cuCtxDestroy and cuCtxDestroy_v2, the last release and a reset of the primary
//     context, by both names of each): the bytes in use in the heap stay as they were over 500
//     contexts, with a module each (an image of 8 kB). A release of the primary context
//     that is not its last, and the destruction of another context, leave the modules as they are
//     (a kernel that the layer refuses is still refused); a library stays when the context in
//     which the driver gave a module for it goes. The cubin of tests/kernels/k_fit_test (CUBIN in
//     the environment), whose original the layer keeps for a second translation at launch, is
//     dropped with its context too.
//   - lazy_lock: a library restored from the layer's cache, whose kernel's code is not in the
//     cache, is adapted when the program asks for the kernel; meanwhile a kernel of another
//     library is given at once (the lock of all lazy libraries was held over the adaptation), and
//     another thread that asks for the same kernel gets the same, once.
//   - cache_temporaries: a write to the layer's cache that fails (a limit on the size of files)
//     leaves no temporary file; the first use of the cache removes a temporary file that a write
//     left an hour before, not one of now.
//
// usage: lifecycle CASE PROXY DIR
// PROXY is the layer's libcuda.so.1, DIR a directory for the files of the case.
#define _GNU_SOURCE
#include <dlfcn.h>
#include <fcntl.h>
#include <malloc.h>
#include <pthread.h>
#include <signal.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <dirent.h>
#include <sys/ioctl.h>
#include <sys/resource.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <sys/time.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>

typedef int CUresult;

static void *lib;
static int checks = 0, wrong = 0;

static void *sym(const char *name) {
    void *p = dlsym(lib, name);
    if (!p) { fprintf(stderr, "missing symbol in proxy: %s\n", name); exit(2); }
    return p;
}
static void open_proxy(const char *path) {
    lib = dlopen(path, RTLD_NOW);
    if (!lib) { fprintf(stderr, "cannot open proxy: %s\n", dlerror()); exit(2); }
}
static void check(const char *name, int ok, const char *what) {
    checks++;
    if (!ok) wrong++;
    printf("%s  %-44s %s\n", ok ? "ok   " : "WRONG", name, what);
}

// PTX for sm_75 with a kernel that the layer refuses (barrier.cluster has no sequence for sm_70)
// and one that it gives.
static const char ptx[] =
    ".version 8.8\n.target sm_75\n.address_size 64\n"
    ".visible .entry bad()\n{\n barrier.cluster.arrive;\n ret;\n}\n"
    ".visible .entry good()\n{\n ret;\n}\n";

// ---- fork_exit ----

static atomic_int stop, loader_tid;

// Loads a module and unloads it, again and again: at VF_LOG_LEVEL=3, lines of log each time.
static void *loader(void *arg) {
    (void)arg;
    CUresult (*load)(void **, const void *) = sym("cuModuleLoadData");
    CUresult (*unload)(void *) = sym("cuModuleUnload");
    loader_tid = (int)syscall(SYS_gettid);
    while (!stop) {
        void *module;
        if (load(&module, ptx) == 0) unload(module);
    }
    return NULL;
}

// Whether the thread `tid` of this process waits in a write().
static int in_write(int tid) {
    char path[64];
    snprintf(path, sizeof path, "/proc/self/task/%d/syscall", tid);
    FILE *f = fopen(path, "r");
    if (!f) return 0;
    long nr = -1;
    int parsed = fscanf(f, "%ld", &nr);
    fclose(f);
    return parsed == 1 && nr == SYS_write;
}

static int fork_exit(const char *proxy, const char *dir) {
    char fifo[4096];
    snprintf(fifo, sizeof fifo, "%s/log.fifo", dir);
    unlink(fifo);
    if (mkfifo(fifo, 0600)) { perror("mkfifo"); return 2; }
    // Open for reading and writing: the layer's open of its log does not wait for a reader, and
    // nothing is read.
    int pipe_fd = open(fifo, O_RDWR);
    if (pipe_fd < 0) { perror("open"); return 2; }
    signal(SIGPIPE, SIG_IGN);
    setenv("VF_LOG", fifo, 1);
    setenv("VF_LOG_LEVEL", "3", 1);
    setenv("VF_STATS", "1", 1);  // the end of the process reports the calls
    open_proxy(proxy);

    pthread_t thread;
    if (pthread_create(&thread, NULL, loader, NULL)) { fprintf(stderr, "fork_exit: no thread\n"); return 2; }
    // Until the pipe is full and the thread waits in a write of the log, with the log's lock: it
    // is in a write, and the pipe has as much in it as 10 ms before.
    int capacity = fcntl(pipe_fd, F_GETPIPE_SZ), queued = 0, waits = 0;
    for (int i = 0; i < 1000 && !waits; i++) {
        int before = queued;
        usleep(10000);
        ioctl(pipe_fd, FIONREAD, &queued);
        waits = queued > capacity / 2 && queued == before && loader_tid && in_write(loader_tid);
    }
    if (!waits) {
        fprintf(stderr, "fork_exit: the thread did not come to wait in a write of the log (%d bytes in the pipe)\n", queued);
        _exit(2);
    }

    fflush(stdout);
    pid_t child = fork();
    if (child == 0) exit(0);
    if (child < 0) { perror("fork"); return 2; }
    int status = 0, ended = 0;
    for (int i = 0; i < 500 && !ended; i++) {
        ended = waitpid(child, &status, WNOHANG) == child;
        if (!ended) usleep(10000);
    }
    if (!ended) {
        kill(child, SIGKILL);
        waitpid(child, &status, 0);
    }
    check("fork_exit_child_ends", ended && WIFEXITED(status) && WEXITSTATUS(status) == 0,
          ended ? "the child ended" : "the child did not end in 5 s");

    // Let the thread go: what it writes is read until it is done.
    stop = 1;
    fcntl(pipe_fd, F_SETFL, fcntl(pipe_fd, F_GETFL) | O_NONBLOCK);
    char buffer[65536];
    while (pthread_tryjoin_np(thread, NULL)) {
        while (read(pipe_fd, buffer, sizeof buffer) > 0) {}
        usleep(1000);
    }
    // No reader any more: what the layer writes at the end of this process goes nowhere.
    close(pipe_fd);
    unlink(fifo);
    return 0;
}

// ---- contexts ----

#define CUDA_ERROR_INVALID_PTX 218
#define CYCLES 500
// A second translation takes a twentieth of a second.
#define SECOND_CYCLES 40
// What may come and go in the heap over cycles that keep nothing.
#define NOISE (256 * 1024)

#define SYM(ret, name, ...) static ret (*name)(__VA_ARGS__)
SYM(CUresult, cuCtxCreate_v2, void **, unsigned, int);
SYM(CUresult, cuCtxDestroy, void *);
SYM(CUresult, cuCtxDestroy_v2, void *);
SYM(CUresult, cuCtxSetCurrent, void *);
SYM(CUresult, cuDevicePrimaryCtxRetain, void **, int);
SYM(CUresult, cuDevicePrimaryCtxRelease, int);
SYM(CUresult, cuDevicePrimaryCtxRelease_v2, int);
SYM(CUresult, cuDevicePrimaryCtxReset, int);
SYM(CUresult, cuDevicePrimaryCtxReset_v2, int);
SYM(CUresult, cuModuleLoadData, void **, const void *);
SYM(CUresult, cuModuleGetFunction, void **, void *, const char *);
SYM(CUresult, cuLaunchKernel, void *, unsigned, unsigned, unsigned, unsigned, unsigned, unsigned, unsigned, void *,
    void **, void **);
SYM(CUresult, cuLibraryLoadData, void **, const void *, int *, void **, unsigned, int *, void **, unsigned);
SYM(CUresult, cuLibraryGetKernel, void **, void *, const char *);
SYM(CUresult, cuLibraryGetModule, void **, void *);
SYM(CUresult, cuKernelGetFunction, void **, void *);
SYM(CUresult, cuLibraryUnload, void *);
#define LOAD(name) *(void **)&name = sym(#name)

// The bytes in use in the heap, which the layer's allocations are in, with the blocks that malloc
// maps on their own (a large table).
static long heap(void) {
    struct mallinfo2 info = mallinfo2();
    return (long)(info.uordblks + info.hblkhd);
}

static void check_growth(const char *name, long before, int cycles) {
    long growth = heap() - before;
    char what[96];
    snprintf(what, sizeof what, "heap %+ld bytes over %d contexts (want less than %d)", growth, cycles, NOISE);
    check(name, growth < NOISE, what);
}

// The image of `ptx`, with a comment that makes it as large as an image of a program.
#define COMMENT 8192
static char *large_image(void) {
    char *image = malloc(sizeof ptx + 3 + COMMENT + 1);
    strcpy(image, ptx);
    strcat(image, "// ");
    memset(image + strlen(image), 'x', COMMENT);
    strcpy(image + sizeof ptx - 1 + 3 + COMMENT, "\n");
    return image;
}

static void *read_file(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "cannot open %s\n", path); exit(2); }
    fseek(f, 0, SEEK_END);
    long size = ftell(f);
    fseek(f, 0, SEEK_SET);
    void *data = malloc(size);
    if (fread(data, 1, size, f) != (size_t)size) { fprintf(stderr, "cannot read %s\n", path); exit(2); }
    fclose(f);
    return data;
}

static int contexts(const char *proxy, const char *dir) {
    // Translations kept: a translation is taken from there after the first.
    char cache[4096];
    snprintf(cache, sizeof cache, "%s/cache", dir);
    if (mkdir(cache, 0700)) { perror("mkdir"); return 2; }
    setenv("VF_PROXY_CACHE", cache, 1);
    open_proxy(proxy);
    LOAD(cuCtxCreate_v2); LOAD(cuCtxDestroy); LOAD(cuCtxDestroy_v2); LOAD(cuCtxSetCurrent);
    LOAD(cuDevicePrimaryCtxRetain); LOAD(cuDevicePrimaryCtxRelease); LOAD(cuDevicePrimaryCtxRelease_v2);
    LOAD(cuDevicePrimaryCtxReset); LOAD(cuDevicePrimaryCtxReset_v2);
    LOAD(cuModuleLoadData); LOAD(cuModuleGetFunction); LOAD(cuLaunchKernel);
    LOAD(cuLibraryLoadData); LOAD(cuLibraryGetKernel); LOAD(cuLibraryGetModule); LOAD(cuLibraryUnload);
    LOAD(cuKernelGetFunction);
    char *image = large_image();
    void *context, *other, *module, *function, *library, *kernel;
    long before = 0;
    char what[96];

    // A context made and destroyed, with a module: by either name of the call.
    for (int i = 0; i < CYCLES + 50; i++) {
        if (i == 50) before = heap();
        cuCtxCreate_v2(&context, 0, 0);
        cuModuleLoadData(&module, image);
        cuModuleGetFunction(&function, module, "good");
        (i & 1 ? cuCtxDestroy : cuCtxDestroy_v2)(context);
    }
    check_growth("context_destroyed", before, CYCLES);

    // The primary context retained twice: the first release leaves the module as it is (its
    // refused kernel is still refused), the second destroys the context.
    int still_refused = 0;
    for (int i = 0; i < CYCLES + 50; i++) {
        if (i == 50) before = heap();
        cuDevicePrimaryCtxRetain(&context, 0);
        cuDevicePrimaryCtxRetain(&context, 0);
        cuCtxSetCurrent(context);
        cuModuleLoadData(&module, image);
        (i & 1 ? cuDevicePrimaryCtxRelease : cuDevicePrimaryCtxRelease_v2)(0);
        still_refused += cuModuleGetFunction(&function, module, "bad") == CUDA_ERROR_INVALID_PTX;
        (i & 1 ? cuDevicePrimaryCtxRelease_v2 : cuDevicePrimaryCtxRelease)(0);
    }
    snprintf(what, sizeof what, "refused after the first release %d times of %d", still_refused, CYCLES + 50);
    check("primary_not_last_release", still_refused == CYCLES + 50, what);
    check_growth("primary_last_release", before, CYCLES);

    // The primary context reset while it is retained: its modules go.
    cuDevicePrimaryCtxRetain(&context, 0);
    cuCtxSetCurrent(context);
    for (int i = 0; i < CYCLES + 50; i++) {
        if (i == 50) before = heap();
        cuModuleLoadData(&module, image);
        cuModuleGetFunction(&function, module, "good");
        (i & 1 ? cuDevicePrimaryCtxReset : cuDevicePrimaryCtxReset_v2)(0);
    }
    check_growth("primary_reset", before, CYCLES);
    cuDevicePrimaryCtxRelease_v2(0);

    // Another context destroyed: the module of this one stays as it is.
    cuCtxCreate_v2(&context, 0, 0);
    cuModuleLoadData(&module, image);
    cuCtxCreate_v2(&other, 0, 0);
    cuCtxDestroy_v2(other);
    cuCtxSetCurrent(context);
    check("other_context_destroyed", cuModuleGetFunction(&function, module, "bad") == CUDA_ERROR_INVALID_PTX,
          "the refused kernel of this context's module is refused");
    cuCtxDestroy_v2(context);

    // A library, and the module that the driver gives for it in a context that goes: the library
    // stays as it is.
    cuCtxCreate_v2(&context, 0, 0);
    cuLibraryLoadData(&library, image, NULL, NULL, 0, NULL, NULL, 0);
    cuLibraryGetModule(&module, library);
    cuCtxDestroy_v2(context);
    cuCtxCreate_v2(&context, 0, 0);
    check("library_stays",
          cuLibraryGetKernel(&kernel, library, "bad") == CUDA_ERROR_INVALID_PTX
              && cuLibraryGetKernel(&kernel, library, "good") == 0,
          "its refused kernel is refused, the other given");
    cuCtxDestroy_v2(context);
    // The module that the driver gives for the library in each context, and a function of it; the
    // function of the library's kernel there, which leads to that module too.
    cuLibraryGetKernel(&kernel, library, "good");
    for (int i = 0; i < 10 * CYCLES + 50; i++) {
        if (i == 50) before = heap();
        cuCtxCreate_v2(&context, 0, 0);
        cuLibraryGetModule(&module, library);
        cuModuleGetFunction(&function, module, "good");
        cuKernelGetFunction(&function, kernel);
        cuCtxDestroy_v2(context);
    }
    check_growth("context_destroyed_library_module", before, 10 * CYCLES);
    cuLibraryUnload(library);

    // A translated cubin whose original the layer keeps for a second translation at launch.
    const char *cubin_path = getenv("CUBIN");
    if (!cubin_path) { fprintf(stderr, "contexts: CUBIN is not set\n"); return 2; }
    void *cubin = read_file(cubin_path);
    for (int i = 0; i < CYCLES + 50; i++) {
        if (i == 50) before = heap();
        cuCtxCreate_v2(&context, 0, 0);
        cuModuleLoadData(&module, cubin);
        cuModuleGetFunction(&function, module, "k_fit_regs");
        cuCtxDestroy_v2(context);
    }
    check_growth("context_destroyed_translated", before, CYCLES);
    // And launched from a second translation, a module of the same context: the stand-in refuses
    // the launches of the first module of a context.
    int launched = 0;
    for (int i = 0; i < SECOND_CYCLES + 10; i++) {
        if (i == 10) before = heap();
        cuCtxCreate_v2(&context, 0, 0);
        cuModuleLoadData(&module, cubin);
        cuModuleGetFunction(&function, module, "k_fit_regs");
        launched += cuLaunchKernel(function, 1, 1, 1, 1024, 1, 1, 0, NULL, NULL, NULL) == 0;
        cuCtxDestroy_v2(context);
    }
    snprintf(what, sizeof what, "%d launches of %d", launched, SECOND_CYCLES + 10);
    check("second_translation_launched", launched == SECOND_CYCLES + 10, what);
    check_growth("context_destroyed_second_translation", before, SECOND_CYCLES);
    return 0;
}

// ---- lazy_lock ----

static double now(void) {
    struct timeval t;
    gettimeofday(&t, NULL);
    return t.tv_sec + t.tv_usec / 1e6;
}

// PTX for sm_75 with many kernels, whose adaptation takes a while.
static char *many_kernels(int kernels, int lines) {
    size_t size = 128 + (size_t)kernels * (160 + (size_t)lines * 40);
    char *image = malloc(size), *at = image;
    at += sprintf(at, ".version 8.8\n.target sm_75\n.address_size 64\n");
    for (int k = 0; k < kernels; k++) {
        at += sprintf(at, ".visible .entry k%d(.param .u64 p)\n{\n .reg .b32 %%r<4>;\n .reg .b64 %%rd<2>;\n"
                          " ld.param.u64 %%rd1, [p];\n mov.u32 %%r1, %%tid.x;\n", k);
        for (int i = 0; i < lines; i++) at += sprintf(at, " add.s32 %%r1, %%r1, %d;\n", i + k);
        at += sprintf(at, " st.global.u32 [%%rd1], %%r1;\n ret;\n}\n");
    }
    return image;
}

struct lookup { void *library; const char *name; void *kernel; CUresult result; double start, end; };
static void *lookup_thread(void *arg) {
    struct lookup *l = arg;
    l->start = now();
    l->result = cuLibraryGetKernel(&l->kernel, l->library, l->name);
    l->end = now();
    return NULL;
}

// Whether the file at `path` has `text` in it.
static int file_has(const char *path, const char *text) {
    FILE *f = fopen(path, "r");
    if (!f) return 0;
    char line[4096];
    int found = 0;
    while (!found && fgets(line, sizeof line, f)) found = strstr(line, text) != NULL;
    fclose(f);
    return found;
}

static int lazy_lock(const char *proxy, const char *dir) {
    char cache[4096], log[4096];
    snprintf(cache, sizeof cache, "%s/cache", dir);
    snprintf(log, sizeof log, "%s/log", dir);
    if (mkdir(cache, 0700)) { perror("mkdir"); return 2; }
    setenv("VF_PROXY_CACHE", cache, 1);
    setenv("VF_LOG", log, 1);
    setenv("VF_LOG_LEVEL", "2", 1);
    open_proxy(proxy);
    LOAD(cuLibraryLoadData); LOAD(cuLibraryGetKernel); LOAD(cuLibraryUnload);
    char *large = many_kernels(400, 200);
    void *first, *other, *kernel;
    // Loaded once and unloaded: the layer keeps what it found of the library, not the code of its
    // kernels. Loaded again: restored from there.
    if (cuLibraryLoadData(&first, large, NULL, NULL, 0, NULL, NULL, 0)) return 2;
    cuLibraryUnload(first);
    if (cuLibraryLoadData(&first, large, NULL, NULL, 0, NULL, NULL, 0)) return 2;
    if (cuLibraryLoadData(&other, ptx, NULL, NULL, 0, NULL, NULL, 0)) return 2;
    if (cuLibraryGetKernel(&kernel, other, "good")) return 2;

    struct lookup slow = { .library = first, .name = "k7" }, same = { .library = first, .name = "k7" },
                  fast = { .library = other, .name = "good" };
    pthread_t thread, again;
    if (pthread_create(&thread, NULL, lookup_thread, &slow)) return 2;
    int adapting = 0;
    for (int i = 0; i < 10000 && !adapting && !slow.end; i++) {
        adapting = file_has(log, "lazy: kernel k7 not cached, adapting original on demand");
        if (!adapting) usleep(1000);
    }
    lookup_thread(&fast);
    if (pthread_create(&again, NULL, lookup_thread, &same)) return 2;
    pthread_join(thread, NULL);
    pthread_join(again, NULL);
    if (!adapting) { fprintf(stderr, "lazy_lock: the library was not adapted on demand\n"); return 2; }
    double took = slow.end - slow.start, waited = fast.end - fast.start;
    char what[128];
    snprintf(what, sizeof what, "%.1f ms, while the other took %.1f ms (want less than a quarter)", waited * 1e3,
             took * 1e3);
    check("lazy_other_library_during_adaptation", !slow.result && !fast.result && waited < took / 4, what);
    snprintf(what, sizeof what, "error %d, kernel %p (want 0, %p)", same.result, same.kernel, slow.kernel);
    check("lazy_same_kernel_during_adaptation", !same.result && same.kernel == slow.kernel, what);
    return 0;
}

// ---- cache_temporaries ----

// The temporary files in `dir` other than the two that the case made.
static int temporaries_of_the_layer(const char *dir) {
    DIR *d = opendir(dir);
    if (!d) return -1;
    int count = 0;
    for (struct dirent *e; (e = readdir(d));)
        count += !strncmp(e->d_name, ".tmp-", 5) && strcmp(e->d_name, ".tmp-00000000000000aa")
                 && strcmp(e->d_name, ".tmp-00000000000000bb");
    closedir(d);
    return count;
}

static int cache_temporaries(const char *proxy, const char *dir) {
    char cache[4096], old[4200], fresh[4200];
    snprintf(cache, sizeof cache, "%s/cache", dir);
    snprintf(old, sizeof old, "%s/.tmp-00000000000000aa", cache);
    snprintf(fresh, sizeof fresh, "%s/.tmp-00000000000000bb", cache);
    if (mkdir(cache, 0700)) { perror("mkdir"); return 2; }
    // Left by writes that did not end, an hour ago and now.
    for (const char *path = old; path; path = path == old ? fresh : NULL) {
        FILE *f = fopen(path, "w");
        if (!f) { perror("fopen"); return 2; }
        fputs("left", f);
        fclose(f);
    }
    struct timeval hour_ago[2] = { { .tv_sec = time(NULL) - 3600 }, { .tv_sec = time(NULL) - 3600 } };
    if (utimes(old, hour_ago)) { perror("utimes"); return 2; }
    setenv("VF_PROXY_CACHE", cache, 1);
    open_proxy(proxy);
    LOAD(cuLibraryLoadData); LOAD(cuLibraryUnload);

    // The layer writes what it found of the library to its cache: more than the limit.
    struct rlimit limit;
    getrlimit(RLIMIT_FSIZE, &limit);
    struct rlimit small = { .rlim_cur = 100, .rlim_max = limit.rlim_max };
    signal(SIGXFSZ, SIG_IGN);
    if (setrlimit(RLIMIT_FSIZE, &small)) { perror("setrlimit"); return 2; }
    void *library;
    CUresult loaded = cuLibraryLoadData(&library, ptx, NULL, NULL, 0, NULL, NULL, 0);
    setrlimit(RLIMIT_FSIZE, &limit);
    if (loaded) { fprintf(stderr, "cache_temporaries: load failed (%d)\n", loaded); return 2; }
    cuLibraryUnload(library);

    char what[96];
    int left = temporaries_of_the_layer(cache);
    snprintf(what, sizeof what, "%d temporary files left by the layer (want 0)", left);
    check("cache_failed_write", left == 0, what);
    int gone = access(old, F_OK) != 0, kept = access(fresh, F_OK) == 0;
    check("cache_stale_temporary_removed", gone, gone ? "the one of an hour ago is gone" : "the one of an hour ago is there");
    check("cache_fresh_temporary_kept", kept, kept ? "the one of now is there" : "the one of now is gone");
    return 0;
}

int main(int argc, char **argv) {
    if (argc != 4) { fprintf(stderr, "usage: lifecycle CASE PROXY DIR\n"); return 2; }
    const char *name = argv[1], *proxy = argv[2], *dir = argv[3];
    int status;
    if (!strcmp(name, "fork_exit")) status = fork_exit(proxy, dir);
    else if (!strcmp(name, "contexts")) status = contexts(proxy, dir);
    else if (!strcmp(name, "lazy_lock")) status = lazy_lock(proxy, dir);
    else if (!strcmp(name, "cache_temporaries")) status = cache_temporaries(proxy, dir);
    else { fprintf(stderr, "unknown case: %s\n", name); return 2; }
    if (status) return status;
    printf("%s: %d checks, %d wrong\n", name, checks, wrong);
    return wrong ? 1 : 0;
}
