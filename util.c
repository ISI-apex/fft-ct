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

#include "util.h"

static double dbl_rand(void)
{
    // random number in range [-0.5, 0.5] - this is what FFTW's benchfft does
    double d = rand();
    return (d / (double) RAND_MAX) - 0.5;
}

void complex_fill_rand(fftw_complex *a, size_t len)
{
    size_t i;
    for (i = 0; i < len; i++) {
        a[i][0] = dbl_rand(); // re
        a[i][1] = dbl_rand(); // im
    }
}

void matrix_print(fftw_complex *A, size_t nrows, size_t ncols)
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

static int is_dbl_eq(double a, double b)
{
    double v = a - b;
    return v >= 0.0 ? (v < DBL_EPSILON) : (v > -DBL_EPSILON);
}

int is_complex_eq(const fftw_complex a, const fftw_complex b)
{
    return is_dbl_eq(a[0], b[0]) && is_dbl_eq(a[1], b[1]);
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
