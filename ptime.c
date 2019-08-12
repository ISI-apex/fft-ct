/*
 * Some various functions for dealing with time.
 *
 * These are meant to be as portable as possible, though "struct timespec" must be defined.
 * _GNU_SOURCE should be defined before the first inclusion of time.h.
 *
 * Many thanks to online sources, like:
 *   http://stackoverflow.com/questions/5167269/clock-gettime-alternative-in-mac-os-x
 *   https://janus.conf.meetecho.com/docs/mach__gettime_8h_source.html
 *   http://nadeausoftware.com/articles/2012/04/c_c_tip_how_measure_elapsed_real_time_benchmarking
 *   http://stackoverflow.com/questions/5404277/porting-clock-gettime-to-windows
 *   http://stackoverflow.com/questions/5801813/c-usleep-is-obsolete-workarounds-for-windows-mingw
 *
 * @author Connor Imes
 * @date 2017-02-01
 */
#define _GNU_SOURCE
#include <errno.h>
#include <inttypes.h>
#include <stdlib.h>
#include <time.h>
#include "ptime.h"

/* begin platform-specific headers and definitions */
#if defined(__MACH__)

#include <mach/clock.h>
#include <mach/mach.h>

#elif defined(_WIN32)

#include <Windows.h>

#else

static const clockid_t PTIME_CLOCKID_T_MONOTONIC =
#if defined(CLOCK_MONOTONIC_PRECISE)
  // BSD
  CLOCK_MONOTONIC_PRECISE
// #elif defined(CLOCK_MONOTONIC_RAW)
//   // Linux
//   CLOCK_MONOTONIC_RAW
#elif defined(CLOCK_HIGHRES)
  // Solaris
  CLOCK_HIGHRES;
#elif defined(CLOCK_MONOTONIC)
  // AIX, BSD, Linux, POSIX, Solaris
  CLOCK_MONOTONIC
#else
  #error "No monotonic clock found"
#endif
;

#endif
/* end platform-specific headers and definitions */

#define ONE_THOUSAND 1000
#define ONE_MILLION  1000000
#define ONE_BILLION  1000000000

#if defined(__MACH__)
static int gettime_monotonic_mach(struct timespec *ts) {
  clock_serv_t cclock;
  mach_timespec_t mts;
  int ret;
  host_get_clock_service(mach_host_self(), SYSTEM_CLOCK, &cclock);
  ret = clock_get_time(cclock, &mts);
  mach_port_deallocate(mach_task_self(), cclock);
  if (!ret) {
    ts->tv_sec = mts.tv_sec;
    ts->tv_nsec = mts.tv_nsec;
  }
  return ret;
}
#endif // __MACH__

#if defined(_WIN32)
static int gettime_monotonic_win32(struct timespec *ts) {
  static LONG g_first_time = 1;
  static LARGE_INTEGER g_counts_per_sec;
  LARGE_INTEGER count;
  // TODO: thread-safe initializer
  if (g_first_time) {
    if (QueryPerformanceFrequency(&g_counts_per_sec) == 0) {
      g_counts_per_sec.QuadPart = 0;
    }
    g_first_time = 0;
  }
  if (g_counts_per_sec.QuadPart <= 0 || QueryPerformanceCounter(&count) == 0) {
    return -1;
  }
  ts->tv_sec = count.QuadPart / g_counts_per_sec.QuadPart;
  ts->tv_nsec = ((count.QuadPart % g_counts_per_sec.QuadPart) * 1E9) / g_counts_per_sec.QuadPart;
  return 0;
}
#endif // _WIN32

int ptime_gettime_monotonic(struct timespec *ts) {
#if defined(__MACH__)
  return gettime_monotonic_mach(ts);
#elif defined(_WIN32)
  return gettime_monotonic_win32(ts);
#else
  return clock_gettime(PTIME_CLOCKID_T_MONOTONIC, ts);
#endif
}
