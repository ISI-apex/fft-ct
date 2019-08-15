/*
 * Some various functions for dealing with time.
 *
 * @author Connor Imes
 * @date 2017-02-01
 */
#ifndef _PTIME_H_
#define _PTIME_H_

#ifdef __cplusplus
extern "C" {
#endif

#define _GNU_SOURCE
#include <inttypes.h>
#include <time.h>

#pragma GCC visibility push(hidden)

int ptime_gettime_monotonic(struct timespec *ts);

int64_t ptime_elapsed_ns(const struct timespec *t1, const struct timespec *t2);

int64_t ptime_elapsed_us(const struct timespec *t1, const struct timespec *t2);

int64_t ptime_elapsed_ms(const struct timespec *t1, const struct timespec *t2);

#pragma GCC visibility pop

#ifdef __cplusplus
}
#endif

#endif
