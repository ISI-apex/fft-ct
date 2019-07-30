/**
 * Transpose functions.
 *
 * @author Connor Imes <cimes@isi.edu>
 * @date 2019-07-15 
 */
#ifndef TRANSPOSE_H
#define TRANSPOSE_H

#include <stdlib.h>

#include <fftw3.h>

#include <mkl.h>

void transpose_dbl_naive(const double* restrict A, double* restrict B,
			 size_t A_rows, size_t A_cols);

void transpose_fftw_complex_naive(fftw_complex* restrict A, fftw_complex* restrict B,
                                  size_t A_rows, size_t A_cols);

void transpose_dbl_mkl(const double* restrict A, double* restrict B,
                       size_t A_rows, size_t A_cols);

void transpose_cmplx16_mkl(const MKL_Complex16* restrict A, MKL_Complex16* restrict B,
                           size_t A_rows, size_t A_cols);

#endif /* TRANSPOSE_H */
