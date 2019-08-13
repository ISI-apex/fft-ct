/**
 * Utility functions
 *
 * @author Connor Imes <cimes@isi.edu>
 * @date 2019-07-15 
 */
#include <errno.h>
#include <float.h>
#include <stdio.h>
#include <stdlib.h>

#include "util.h"

float rand_flt(void)
{
    // random number in range [-0.5, 0.5] - this is what FFTW's benchfft does
    float d = rand();
    return (d / (float) RAND_MAX) - 0.5;
}

double rand_dbl(void)
{
    // random number in range [-0.5, 0.5] - this is what FFTW's benchfft does
    double d = rand();
    return (d / (double) RAND_MAX) - 0.5;
}

void fill_rand_flt(float *a, size_t len)
{
    size_t i;
    for (i = 0; i < len; i++)
        a[i] = rand_flt();
}

void fill_rand_dbl(double *a, size_t len)
{
    size_t i;
    for (i = 0; i < len; i++)
        a[i] = rand_dbl();
}

void matrix_print_flt(const float *A, size_t nrows, size_t ncols)
{
    size_t r, c;
    for (r = 0; r < nrows; r++) {
        for (c = 0; c < ncols; c++) {
            printf("%s%f", (c > 0 ? ", " : ""), A[r * ncols + c]);
        }
        printf("\n");
    }
}

void matrix_print_dbl(const double *A, size_t nrows, size_t ncols)
{
    size_t r, c;
    for (r = 0; r < nrows; r++) {
        for (c = 0; c < ncols; c++) {
            printf("%s%f", (c > 0 ? ", " : ""), A[r * ncols + c]);
        }
        printf("\n");
    }
}

int is_eq_flt(float a, float b)
{
    float v = a - b;
    return v >= 0.0 ? (v < FLT_EPSILON) : (v > -FLT_EPSILON);
}

int is_eq_dbl(double a, double b)
{
    double v = a - b;
    return v >= 0.0 ? (v < DBL_EPSILON) : (v > -DBL_EPSILON);
}

void *assert_malloc(size_t sz)
{
    void *ptr = malloc(sz);
    if (!ptr) {
        perror("malloc");
        exit(ENOMEM);
    }
    return ptr;
}

void *assert_malloc_al(size_t sz)
{
    size_t align;
    void *ptr;
    if (sz % 64 == 0) {
        align = 64;
    } else if (sz % 32 == 0) {
        align = 32;
    } else {
        fprintf(stderr, "assert_malloc: sz must be a multiple of 64 or 32\n");
        exit(EINVAL);
    }
#if defined(HAVE_ALIGNED_ALLOC)
    ptr = aligned_alloc(align, sz);
    if (!ptr) {
        perror("aligned_alloc");
        exit(ENOMEM);
    }
#else
    errno = posix_memalign(&ptr, align, sz);
    if (errno) {
        perror("posix_memalign");
        exit(ENOMEM);
    }
#endif
    return ptr;
}
