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

#include <fftw3.h>

#include <mkl.h>

#include "util.h"

static float flt_rand(void)
{
    // random number in range [-0.5, 0.5] - this is what FFTW's benchfft does
    float d = rand();
    return (d / (float) RAND_MAX) - 0.5;
}

void fill_rand_flt(float *a, size_t len)
{
    size_t i;
    for (i = 0; i < len; i++)
        a[i] = flt_rand();
}

static double dbl_rand(void)
{
    // random number in range [-0.5, 0.5] - this is what FFTW's benchfft does
    double d = rand();
    return (d / (double) RAND_MAX) - 0.5;
}

void fill_rand_dbl(double *a, size_t len)
{
    size_t i;
    for (i = 0; i < len; i++)
        a[i] = dbl_rand();
}

void fill_rand_fftw_complex(fftw_complex *a, size_t len)
{
    size_t i;
    for (i = 0; i < len; i++) {
        a[i][0] = dbl_rand(); // re
        a[i][1] = dbl_rand(); // im
    }
}

void fill_rand_cmplx16(MKL_Complex16 *a, size_t len)
{
    size_t i;
    for (i = 0; i < len; i++) {
        a[i].real = dbl_rand();
        a[i].imag = dbl_rand();
    }
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

void matrix_print_fftw_complex(fftw_complex *A, size_t nrows, size_t ncols)
{
    size_t r, c, i;
    for (r = 0; r < nrows; r++) {
        for (c = 0; c < ncols; c++) {
            i = r * ncols + c;
            printf("%s(%f, %f)", (c > 0 ? ", " : ""), A[i][0], A[i][1]);
        }
        printf("\n");
    }
}

void matrix_print_cmplx16(const MKL_Complex16 *A, size_t nrows, size_t ncols)
{
    size_t r, c, i;
    for (r = 0; r < nrows; r++) {
        for (c = 0; c < ncols; c++) {
            i = r * ncols + c;
            printf("%s(%f, %f)", (c > 0 ? ", " : ""), A[i].real, A[i].imag);
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

int is_eq_fftw_complex(const fftw_complex a, const fftw_complex b)
{
    return is_eq_dbl(a[0], b[0]) && is_eq_dbl(a[1], b[1]);
}

int is_eq_cmplx16(const MKL_Complex16 a, const MKL_Complex16 b)
{
    return is_eq_dbl(a.real, b.real) && is_eq_dbl(a.imag, b.imag);
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

void *assert_fftw_malloc(size_t sz)
{
    void *ptr = fftw_malloc(sz);
    if (!ptr) {
        perror("fftw_malloc");
        exit(ENOMEM);
    }
    return ptr;
}
