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

#include <mkl.h>

void complex_fill_rand(fftw_complex *a, size_t len);

void dbl_fill_rand(double *a, size_t len);

void cmplx16_fill_rand(MKL_Complex16 *a, size_t len);

void matrix_dbl_print(const double *A, size_t nrows, size_t ncols);

void matrix_print(fftw_complex *A, size_t nrows, size_t ncols);

void matrix_cmplx16_print(const MKL_Complex16 *A, size_t nrows, size_t ncols);

int is_dbl_eq(double a, double b);

int is_complex_eq(const fftw_complex a, const fftw_complex b);

int is_cmplx16_eq(const MKL_Complex16 a, const MKL_Complex16 b);

void *assert_fftw_malloc(size_t sz);

void *assert_malloc(size_t sz);

#endif /* UTIL_H */
