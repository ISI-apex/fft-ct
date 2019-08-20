/**
 * Utility functions
 *
 * @author Connor Imes <cimes@isi.edu>
 * @date 2019-07-15 
 */
#include <complex.h>
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

float complex rand_fcmplx(void)
{
    return CMPLXF(rand_flt(), rand_flt());
}

double complex rand_dcmplx(void)
{
    return CMPLX(rand_dbl(), rand_dbl());
}

#define FILL_RAND(a, len, fn_rand) \
{ \
    size_t i; \
    for (i = 0; i < len; i++) \
        a[i] = fn_rand(); \
}

void fill_rand_flt(float *a, size_t len)
{
    FILL_RAND(a, len, rand_flt);
}

void fill_rand_dbl(double *a, size_t len)
{
    FILL_RAND(a, len, rand_dbl);
}

void fill_rand_fcmplx(float complex *a, size_t len)
{
    FILL_RAND(a, len, rand_fcmplx);
}

void fill_rand_dcmplx(double complex *a, size_t len)
{
    FILL_RAND(a, len, rand_dcmplx);
}

#define MATRIX_PRINT(A, nrows, ncols) \
{ \
    size_t r, c; \
    for (r = 0; r < nrows; r++) { \
        for (c = 0; c < ncols; c++) { \
            printf("%s%f", (c > 0 ? ", " : ""), A[r * ncols + c]); \
        } \
        printf("\n"); \
    } \
}

#define MATRIX_PRINT_CMPLX(A, nrows, ncols) \
{ \
    size_t r, c, i; \
    for (r = 0; r < nrows; r++) { \
        for (c = 0; c < ncols; c++) { \
            i = r * ncols + c; \
            /* TODO: should be crealf and cimagf for complex */ \
            printf("%s(%f, %f)", (c > 0 ? ", " : ""), creal(A[i]), cimag(A[i])); \
        } \
        printf("\n"); \
    } \
}

void matrix_print_flt(const float *A, size_t nrows, size_t ncols)
{
    MATRIX_PRINT(A, nrows, ncols);
}

void matrix_print_dbl(const double *A, size_t nrows, size_t ncols)
{
    MATRIX_PRINT(A, nrows, ncols);
}

void matrix_print_fcmplx(const float complex *A, size_t nrows, size_t ncols)
{
    MATRIX_PRINT_CMPLX(A, nrows, ncols);
}

void matrix_print_dcmplx(const double complex *A, size_t nrows, size_t ncols)
{
    MATRIX_PRINT_CMPLX(A, nrows, ncols);
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

int is_eq_fcmplx(float complex a, float complex b)
{
    return is_eq_flt(crealf(a), crealf(b)) && is_eq_flt(cimagf(a), cimagf(b));
}

int is_eq_dcmplx(double complex a, double complex b)
{
    return is_eq_dbl(creal(a), creal(b)) && is_eq_dbl(cimag(a), cimag(b));
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
