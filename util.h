/**
 * Utility functions
 *
 * @author Connor Imes <cimes@isi.edu>
 * @date 2019-07-15 
 */
#ifndef UTIL_H
#define UTIL_H

#include <stdlib.h>

#include <fftw3.h>

void complex_fill_rand(fftw_complex *a, size_t len);

void matrix_print(fftw_complex *A, size_t nrows, size_t ncols);

int is_complex_eq(const fftw_complex a, const fftw_complex b);

void *assert_fftw_malloc(size_t sz);

#endif /* UTIL_H */
