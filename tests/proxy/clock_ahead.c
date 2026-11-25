// A stand-in for clock_gettime, put in front of the C library's with LD_PRELOAD by
// tests/proxy/summary_fill.py: it puts the time of day (CLOCK_REALTIME and its coarse form) CLOCK_AHEAD
// seconds ahead of the clock that the kernel counts a deadline of CLOCK_REALTIME by, as when the time of
// day is set while a wait is under way. The other clocks, CLOCK_MONOTONIC among them, are the C library's.
#define _GNU_SOURCE
#include <dlfcn.h>
#include <stdlib.h>
#include <time.h>

int clock_gettime(clockid_t clock, struct timespec *t)
{
    static int (*real)(clockid_t, struct timespec *);
    if (!real) *(void **)&real = dlsym(RTLD_NEXT, "clock_gettime");
    const int r = real(clock, t);
    const char *ahead = getenv("CLOCK_AHEAD");
    if (r == 0 && ahead && (clock == CLOCK_REALTIME || clock == CLOCK_REALTIME_COARSE)) t->tv_sec += atol(ahead);
    return r;
}
