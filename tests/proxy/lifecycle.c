// What the layer keeps over the life of a process, and what it does at its end, without a GPU.
//
// A stand-in for the driver (lifecycle_mock_driver.c) gives every module a handle of its own. Each
// case runs in a process of its own, as the layer reads its settings when it is loaded:
//   - fork_exit: a child made by fork() while another thread of the parent waits in a write of
//     the log, and that ends with exit(), ends. (What the layer would report at the child's end
//     is its parent's, and the lock of the log stays held in the child for ever.)
//
// usage: lifecycle CASE PROXY DIR
// PROXY is the layer's libcuda.so.1, DIR a directory for the files of the case.
#define _GNU_SOURCE
#include <dlfcn.h>
#include <fcntl.h>
#include <pthread.h>
#include <signal.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ioctl.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <sys/wait.h>
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

int main(int argc, char **argv) {
    if (argc != 4) { fprintf(stderr, "usage: lifecycle CASE PROXY DIR\n"); return 2; }
    const char *name = argv[1], *proxy = argv[2], *dir = argv[3];
    int status;
    if (!strcmp(name, "fork_exit")) status = fork_exit(proxy, dir);
    else { fprintf(stderr, "unknown case: %s\n", name); return 2; }
    if (status) return status;
    printf("%s: %d checks, %d wrong\n", name, checks, wrong);
    return wrong ? 1 : 0;
}
