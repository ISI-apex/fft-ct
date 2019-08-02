/**
 * Transpose functions.
 *
 * @author Connor Imes <cimes@isi.edu>
 * @date 2019-07-15 
 */
#ifndef TRANSPOSE_MKL_H
#define TRANSPOSE_MKL_H

#include <stdlib.h>

#include <mkl.h>

void transpose_flt_mkl(const float* restrict A, float* restrict B,
                       size_t A_rows, size_t A_cols);

void transpose_dbl_mkl(const double* restrict A, double* restrict B,
                       size_t A_rows, size_t A_cols);

void transpose_cmplx8_mkl(const MKL_Complex8* restrict A,
                          MKL_Complex8* restrict B,
                          size_t A_rows, size_t A_cols);

void transpose_cmplx16_mkl(const MKL_Complex16* restrict A,
                           MKL_Complex16* restrict B,
                           size_t A_rows, size_t A_cols);

#endif /* TRANSPOSE_MKL_H */
