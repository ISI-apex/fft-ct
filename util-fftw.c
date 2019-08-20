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
#include "util-fftw.h"

void fill_rand_fftw(fftw_complex *a, size_t len)
{
    fill_rand_dcmplx(a, len);
}

void matrix_print_fftw(const fftw_complex *A, size_t nrows, size_t ncols)
{
    matrix_print_dcmplx(A, nrows, ncols);
}

int is_eq_fftw(fftw_complex a, fftw_complex b)
{
    return is_eq_dcmplx(a, b);
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
