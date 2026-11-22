// CUPTI capture for the three marked requests in zml.py. Build with CUDA 12.9.
#define _GNU_SOURCE
#include <cupti.h>
#include <inttypes.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/un.h>
#include <unistd.h>

static FILE *trace;
static pthread_mutex_t lock = PTHREAD_MUTEX_INITIALIZER;
static size_t dropped;
static int failed;
static const CUpti_ActivityKind kinds[] = {CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL, CUPTI_ACTIVITY_KIND_DRIVER};

static int ok(CUptiResult result, const char *call) {
    if (result == CUPTI_SUCCESS) return 1;
    const char *message = "unknown";
    cuptiGetResultString(result, &message);
    fprintf(stderr, "LLMD profile: %s: %s\n", call, message);
    return 0;
}

static void CUPTIAPI allocate(uint8_t **buffer, size_t *size, size_t *max_records) {
    *max_records = 0;
    *size = 2 * 1024 * 1024;
    if (posix_memalign((void **)buffer, 16, *size)) {
        *buffer = NULL;
        *size = 0;
    }
}

static void CUPTIAPI consume(CUcontext context, uint32_t stream, uint8_t *buffer,
                             size_t capacity, size_t valid) {
    (void)capacity;
    pthread_mutex_lock(&lock);
    if (trace && valid) {
        CUpti_Activity *record = NULL;
        CUptiResult result;
        while ((result = cuptiActivityGetNextRecord(buffer, valid, &record)) == CUPTI_SUCCESS) {
            if (record->kind == CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL) {
                const CUpti_ActivityKernel9 *k = (const void *)record;
                fprintf(trace, "K\t%"PRIu64"\t%"PRIu64"\t%u\t%u\t%d,%d,%d\t%d,%d,%d\t%s\n",
                        k->start, k->end, k->streamId, k->correlationId, k->gridX, k->gridY, k->gridZ,
                        k->blockX, k->blockY, k->blockZ, k->name ? k->name : "?");
            } else if (record->kind == CUPTI_ACTIVITY_KIND_DRIVER) {
                const CUpti_ActivityAPI *a = (const void *)record;
                if (a->cbid == CUPTI_DRIVER_TRACE_CBID_cuGraphLaunch || a->cbid == CUPTI_DRIVER_TRACE_CBID_cuGraphLaunch_ptsz)
                    fprintf(trace, "A\t%"PRIu64"\t%"PRIu64"\t%u\t%u\t%u\n",
                            a->start, a->end, a->threadId, a->correlationId, a->cbid);
            }
        }
        if (result != CUPTI_ERROR_MAX_LIMIT_REACHED) failed = !ok(result, "read activity records");
    }
    size_t missed = 0;
    if (!ok(cuptiActivityGetNumDroppedRecords(context, stream, &missed), "get dropped records")) failed = 1;
    dropped += missed;
    pthread_mutex_unlock(&lock);
    free(buffer);
}

static int mark(int stage, const char *directory) {
    if (stage == 0) {
        char *path = NULL;
        if (asprintf(&path, "%s/activity.tsv", directory) < 0) return 0;
        trace = fopen(path, "wx");
        free(path);
        if (!trace) { perror("LLMD profile trace"); return 0; }
        if (!ok(cuptiActivityRegisterCallbacks(allocate, consume), "register callbacks")) return 0;
        for (size_t i = 0; i < sizeof(kinds)/sizeof(kinds[0]); ++i)
            if (!ok(cuptiActivityEnable(kinds[i]), "enable activities")) return 0;
    }
    uint64_t timestamp;
    if (!ok(cuptiGetTimestamp(&timestamp), "read GPU clock")) return 0;
    pthread_mutex_lock(&lock);
    fprintf(trace, "P\t%"PRIu64"\t%d\n", timestamp, stage);
    pthread_mutex_unlock(&lock);
    if (stage != 5) return 1;
    if (!ok(cuptiActivityFlushAll(CUPTI_ACTIVITY_FLAG_FLUSH_FORCED), "flush activities")) return 0;
    for (size_t i = 0; i < sizeof(kinds)/sizeof(kinds[0]); ++i)
        if (!ok(cuptiActivityDisable(kinds[i]), "disable activities")) return 0;
    pthread_mutex_lock(&lock);
    fprintf(trace, "N\t%zu\n", dropped);
    int valid = !failed && !dropped && !ferror(trace);
    if (fclose(trace)) valid = 0;
    trace = NULL;
    pthread_mutex_unlock(&lock);
    return valid;
}

static void *serve(void *arg) {
    const char *directory = arg;
    struct sockaddr_un address = {.sun_family = AF_UNIX};
    if (snprintf(address.sun_path, sizeof(address.sun_path), "%s/control", directory) >= (int)sizeof(address.sun_path)) {
        fprintf(stderr, "LLMD profile: directory path too long\n");
        return NULL;
    }
    int listener = socket(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC, 0);
    if (listener < 0) { perror("LLMD profile socket"); return NULL; }
    if (bind(listener, (struct sockaddr *)&address, sizeof(address))) {
        perror("LLMD profile bind");
        close(listener);
        return NULL;
    }
    if (!chmod(address.sun_path, 0600) && !listen(listener, 1)) {
        for (int stage = 0; stage < 6; ++stage) {
            int peer = accept4(listener, NULL, NULL, SOCK_CLOEXEC);
            if (peer < 0) break;
            char command, reply = '!';
            if (read(peer, &command, 1) == 1 && command == '0' + stage && mark(stage, directory))
                reply = command;
            ssize_t sent = send(peer, &reply, 1, MSG_NOSIGNAL);
            close(peer);
            if (reply == '!' || sent != 1) break;
        }
    } else {
        perror("LLMD profile listen");
    }
    close(listener);
    unlink(address.sun_path);
    return NULL;
}

__attribute__((constructor)) static void start(void) {
    const char *dir = getenv("VF_LLMD_PROFILE_DIR");
    if (!dir || !*dir) return;
    char path[4096];
    ssize_t n = readlink("/proc/self/exe", path, sizeof(path)-1);
    if (n < 0 || n == (ssize_t)sizeof(path)-1) return;
    path[n] = '\0';
    const char *name = strrchr(path, '/');
    if (!name || strcmp(name+1, "llmd")) return;
    pthread_t thread;
    if (pthread_create(&thread, NULL, serve, (void *)dir) == 0) pthread_detach(thread);
    else fprintf(stderr, "LLMD profile: failed to start control thread\n");
}
