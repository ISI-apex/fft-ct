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

void fill_rand_fftwf_complex(fftwf_complex *a, size_t len)
{
    fill_rand_flt_cmplx(a, len);
}

void matrix_print_fftwf_complex(const fftwf_complex *A,
                                size_t nrows, size_t ncols)
{
    matrix_print_flt_cmplx(A, nrows, ncols);
}

int is_eq_fftwf_complex(fftwf_complex a, fftwf_complex b)
{
    return is_eq_flt_cmplx(a, b);
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
