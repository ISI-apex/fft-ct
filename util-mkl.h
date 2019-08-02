/**
 * Utility functions
 *
 * @author Connor Imes <cimes@isi.edu>
 * @date 2019-07-15 
 */
#ifndef UTIL_MKL_H
#define UTIL_MKL_H

#include <stdlib.h>

#include <mkl.h>

void fill_rand_cmplx8(MKL_Complex8 *a, size_t len);
void fill_rand_cmplx16(MKL_Complex16 *a, size_t len);

void matrix_print_cmplx8(const MKL_Complex8 *A, size_t nrows, size_t ncols);
void matrix_print_cmplx16(const MKL_Complex16 *A, size_t nrows, size_t ncols);

int is_eq_cmplx8(const MKL_Complex8 a, const MKL_Complex8 b);
int is_eq_cmplx16(const MKL_Complex16 a, const MKL_Complex16 b);

#endif /* UTIL_MKL_H */
