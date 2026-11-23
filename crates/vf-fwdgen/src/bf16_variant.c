
// The bf16 variant of this library (forward/bf16tc/): the functions of vf_icpt_names go to the
// runtime beside the library (libvf_cublas_bf16.so) when VF_CUBLAS_BF16_TC or VF_CUBLAS_BF16_SKINNY
// is set and the runtime loads; otherwise, like every other function, to CUDA 12.9 as in the
// default library. A runtime that cannot be loaded is said once, and everything is forwarded.
HIDDEN void *vf_icpt[VF_NICPT + 1];
extern const char *const vf_icpt_names[];
extern const unsigned vf_icpt_slots[];
static void *runtime;
static int runtime_state; /* 0: not tried yet; 1: the runtime serves; 2: everything is forwarded */
static pthread_mutex_t runtime_lock = PTHREAD_MUTEX_INITIALIZER;

static int switch_set(const char *name) {
    const char *v = getenv(name);
    return v && *v && strcmp(v, "0") != 0;
}

static int runtime_open(void) {
    int s = __atomic_load_n(&runtime_state, __ATOMIC_ACQUIRE);
    if (s) return s;
    pthread_mutex_lock(&runtime_lock);
    if (!(s = runtime_state)) {
        s = 2;
        Dl_info info;
        char path[4096];
        if (!switch_set("VF_CUBLAS_BF16_TC") && !switch_set("VF_CUBLAS_BF16_SKINNY")) {
            // No switch: nothing to load.
        } else if (!dladdr((void *)&runtime_open, &info) || !info.dli_fname) {
            fprintf(stderr, "[volta-forever] %s: cannot find its own file; every function is forwarded to CUDA 12.9\n",
                    VF_LIB);
        } else {
            const char *slash = strrchr(info.dli_fname, '/');
            const int dir = slash ? (int)(slash - info.dli_fname + 1) : 0;
            void *h = NULL;
            if (snprintf(path, sizeof path, "%.*s%s", dir, info.dli_fname, VF_RUNTIME) < (int)sizeof path)
                h = dlopen(path, RTLD_NOW | RTLD_LOCAL);
#ifdef VF_BF16_CONTROL_ABORT
            if (!h) abort();
#endif
            if (!h) {
                fprintf(stderr, "[volta-forever] %s: cannot load %s: %s; every function is forwarded to CUDA 12.9\n",
                        VF_LIB, path, dlerror());
            } else {
                int (*init)(void) = (int (*)(void))dlsym(h, "vf_bf16_init");
                if (init && init()) {
                    runtime = h;
                    s = 1;
                }
            }
        }
        __atomic_store_n(&runtime_state, s, __ATOMIC_RELEASE);
    }
    pthread_mutex_unlock(&runtime_lock);
    return s;
}

// First call of the intercepted function `j`: the runtime's, or 12.9's.
HIDDEN void *vf_icpt_resolve(unsigned long j) {
    void *p = runtime_open() == 1 ? dlsym(runtime, vf_icpt_names[j]) : NULL;
    if (!p) p = vf_fwd_resolve(vf_icpt_slots[j]);
    __atomic_store_n(&vf_icpt[j], p, __ATOMIC_RELEASE);
    return p;
}
