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

void fill_rand_flt(float *a, size_t len);
void fill_rand_dbl(double *a, size_t len);
void fill_rand_fftw_complex(fftw_complex *a, size_t len);
void fill_rand_cmplx16(MKL_Complex16 *a, size_t len);

void matrix_print_flt(const float *A, size_t nrows, size_t ncols);
void matrix_print_dbl(const double *A, size_t nrows, size_t ncols);
void matrix_print_fftw_complex(fftw_complex *A, size_t nrows, size_t ncols);
void matrix_print_cmplx16(const MKL_Complex16 *A, size_t nrows, size_t ncols);

int is_eq_flt(float a, float b);
int is_eq_dbl(double a, double b);
int is_eq_fftw_complex(const fftw_complex a, const fftw_complex b);
int is_eq_cmplx16(const MKL_Complex16 a, const MKL_Complex16 b);

void *assert_malloc(size_t sz);
void *assert_fftw_malloc(size_t sz);

#endif /* UTIL_H */
