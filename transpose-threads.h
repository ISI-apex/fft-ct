/**
 * Transpose functions.
 *
 * @author Kaushik Datta <kdatta@isi.edu>
 * @date 2019-08-06
 */
#ifndef TRANSPOSE_THREADS_H
#define TRANSPOSE_THREADS_H

#include <stdlib.h>

void transpose_flt_threads_row(const float* restrict A, float* restrict B,
                               size_t A_rows, size_t A_cols,
                               size_t num_thr);

void transpose_dbl_threads_row(const double* restrict A, double* restrict B,
                               size_t A_rows, size_t A_cols,
                               size_t num_thr);

void transpose_flt_threads_col(const float* restrict A, float* restrict B,
                               size_t A_rows, size_t A_cols,
                               size_t num_thr);

void transpose_dbl_threads_col(const double* restrict A, double* restrict B,
                               size_t A_rows, size_t A_cols,
                               size_t num_thr);

void transpose_flt_cmplx_threads_row(const float complex* restrict A,
                                     float complex* restrict B,
                                     size_t A_rows, size_t A_cols,
                                     size_t num_thr);

void transpose_dbl_cmplx_threads_row(const double complex* restrict A,
                                     double complex* restrict B,
                                     size_t A_rows, size_t A_cols,
                                     size_t num_thr);

void transpose_flt_cmplx_threads_col(const float complex* restrict A,
                                     float complex* restrict B,
                                     size_t A_rows, size_t A_cols,
                                     size_t num_thr);

void transpose_dbl_cmplx_threads_col(const double complex* restrict A,
                                     double complex* restrict B,
                                     size_t A_rows, size_t A_cols,
                                     size_t num_thr);

void transpose_flt_threads_row_blocked(const float* restrict A,
                                       float* restrict B,
                                       size_t A_rows, size_t A_cols,
                                       size_t num_thr,
                                       size_t blk_rows, size_t blk_cols);

void transpose_dbl_threads_row_blocked(const double* restrict A,
                                       double* restrict B,
                                       size_t A_rows, size_t A_cols,
                                       size_t num_thr,
                                       size_t blk_rows, size_t blk_cols);

void transpose_flt_threads_col_blocked(const float* restrict A,
                                       float* restrict B,
                                       size_t A_rows, size_t A_cols,
                                       size_t num_thr,
                                       size_t blk_rows, size_t blk_cols);

void transpose_dbl_threads_col_blocked(const double* restrict A,
                                       double* restrict B,
                                       size_t A_rows, size_t A_cols,
                                       size_t num_thr,
                                       size_t blk_rows, size_t blk_cols);

void transpose_flt_cmplx_threads_row_blocked(const float complex* restrict A,
                                             float complex* restrict B,
                                             size_t A_rows, size_t A_cols,
                                             size_t num_thr,
                                             size_t blk_rows, size_t blk_cols);

void transpose_dbl_cmplx_threads_row_blocked(const double complex* restrict A,
                                             double complex* restrict B,
                                             size_t A_rows, size_t A_cols,
                                             size_t num_thr,
                                             size_t blk_rows, size_t blk_cols);

void transpose_flt_cmplx_threads_col_blocked(const float complex* restrict A,
                                             float complex* restrict B,
                                             size_t A_rows, size_t A_cols,
                                             size_t num_thr,
                                             size_t blk_rows, size_t blk_cols);

void transpose_dbl_cmplx_threads_col_blocked(const double complex* restrict A,
                                             double complex* restrict B,
                                             size_t A_rows, size_t A_cols,
                                             size_t num_thr,
                                             size_t blk_rows, size_t blk_cols);

#endif /* TRANSPOSE_THREADS_H */
