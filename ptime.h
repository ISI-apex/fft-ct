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
#include <time.h>

#pragma GCC visibility push(hidden)

int ptime_gettime_monotonic(struct timespec *ts);

#pragma GCC visibility pop

#ifdef __cplusplus
}
#endif

#endif
