/**
 * Utility functions
 *
 * @author Connor Imes <cimes@isi.edu>
 * @date 2019-07-15 
 */
#ifndef UTIL_FFTW_H
#define UTIL_FFTW_H

#include <stdlib.h>

#include <fftw3.h>

void fill_rand_fftw_complex(fftw_complex *a, size_t len);

void matrix_print_fftw_complex(fftw_complex *A, size_t nrows, size_t ncols);

int is_eq_fftw_complex(const fftw_complex a, const fftw_complex b);

void *assert_fftw_malloc(size_t sz);

#endif /* UTIL_FFTW_H */
