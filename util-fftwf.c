/**
 * Utility functions
 *
 * @author Connor Imes <cimes@isi.edu>
 * @date 2019-07-15 
 */
#include <complex.h>
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>

#include <fftw3.h>

#include "util.h"
#include "util-fftwf.h"

void fill_rand_fftwf(fftwf_complex *a, size_t len)
{
    fill_rand_fcmplx(a, len);
}

void matrix_print_fftwf(const fftwf_complex *A, size_t nrows, size_t ncols)
{
    matrix_print_fcmplx(A, nrows, ncols);
}

int is_eq_fftwf(fftwf_complex a, fftwf_complex b)
{
    return is_eq_fcmplx(a, b);
}

void *assert_fftwf_malloc(size_t sz)
{
    void *ptr = fftwf_malloc(sz);
    if (!ptr) {
        perror("fftwf_malloc");
        exit(ENOMEM);
    }
    return ptr;
}
