/**
 * Utility functions
 *
 * @author Connor Imes <cimes@isi.edu>
 * @date 2019-07-15 
 */
#ifndef UTIL_H
#define UTIL_H

#include <complex.h>
#include <stdlib.h>

float rand_flt(void);
double rand_dbl(void);
float complex rand_flt_cmplx(void);
double complex rand_dbl_cmplx(void);

void fill_rand_flt(float *a, size_t len);
void fill_rand_dbl(double *a, size_t len);
void fill_rand_flt_cmplx(float complex *a, size_t len);
void fill_rand_dbl_cmplx(double complex *a, size_t len);

void matrix_print_flt(const float *A, size_t nrows, size_t ncols);
void matrix_print_dbl(const double *A, size_t nrows, size_t ncols);
void matrix_print_flt_cmplx(const float complex *A, size_t nrows, size_t ncols);
void matrix_print_dbl_cmplx(const double complex *A, size_t nrows, size_t ncols);

int is_eq_flt(float a, float b);
int is_eq_dbl(double a, double b);
int is_eq_flt_cmplx(float complex a, float complex b);
int is_eq_dbl_cmplx(double complex a, double complex b);

void *assert_malloc(size_t sz);
void *assert_malloc_al(size_t sz);

#endif /* UTIL_H */
