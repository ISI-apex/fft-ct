/**
 * Transpose functions.
 *
 * @author Connor Imes <cimes@isi.edu>
 * @author Kaushik Datta <kdatta@isi.edu>
 * @date 2019-07-15
 */
#ifndef TRANSPOSE_H
#define TRANSPOSE_H

#include <complex.h>
#include <stdlib.h>

void transpose_flt_naive(const float* restrict A, float* restrict B,
                         size_t A_rows, size_t A_cols);
void transpose_dbl_naive(const double* restrict A, double* restrict B,
                         size_t A_rows, size_t A_cols);
void transpose_flt_cmplx_naive(const float complex* restrict A,
                               float complex* restrict B,
                               size_t A_rows, size_t A_cols);
void transpose_dbl_cmplx_naive(const double complex* restrict A,
                               double complex* restrict B,
                               size_t A_rows, size_t A_cols);

void transpose_flt_blocked(const float* restrict A, float* restrict B,
                           size_t A_rows, size_t A_cols,
                           size_t blk_rows, size_t blk_cols);
void transpose_dbl_blocked(const double* restrict A, double* restrict B,
                           size_t A_rows, size_t A_cols,
                           size_t blk_rows, size_t blk_cols);
void transpose_flt_cmplx_blocked(const float complex* restrict A,
                                 float complex* restrict B,
                                 size_t A_rows, size_t A_cols,
                                 size_t blk_rows, size_t blk_cols);
void transpose_dbl_cmplx_blocked(const double complex* restrict A,
                                 double complex* restrict B,
                                 size_t A_rows, size_t A_cols,
                                 size_t blk_rows, size_t blk_cols);

#endif /* TRANSPOSE_H */
