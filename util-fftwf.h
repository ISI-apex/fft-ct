/**
 * Utility functions
 *
 * @author Connor Imes <cimes@isi.edu>
 * @date 2019-07-15 
 */
#ifndef UTIL_FFTWF_H
#define UTIL_FFTWF_H

#include <complex.h>
#include <stdlib.h>

#include <fftw3.h>

void fill_rand_fftwf(fftwf_complex *a, size_t len);

void matrix_print_fftwf(const fftwf_complex *A, size_t nrows, size_t ncols);

int is_eq_fftwf(fftwf_complex a, fftwf_complex b);

void *assert_fftwf_malloc(size_t sz);

#endif /* UTIL_FFTWF_H */
