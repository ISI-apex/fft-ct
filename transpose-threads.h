/**
 * Transpose functions.
 *
 * @author Kaushik Datta <kdatta@isi.edu>
 * @date 2019-08-06
 */
#ifndef TRANSPOSE_THREADS_H
#define TRANSPOSE_THREADS_H

#include <complex.h>
#include <stdlib.h>

void transpose_flt_thrrow(const float* restrict A, float* restrict B,
                          size_t A_rows, size_t A_cols,
                          size_t num_thr);
void transpose_dbl_thrrow(const double* restrict A, double* restrict B,
                          size_t A_rows, size_t A_cols,
                          size_t num_thr);
void transpose_fcmplx_thrrow(const float complex* restrict A,
                             float complex* restrict B,
                             size_t A_rows, size_t A_cols,
                             size_t num_thr);
void transpose_dcmplx_thrrow(const double complex* restrict A,
                             double complex* restrict B,
                             size_t A_rows, size_t A_cols,
                             size_t num_thr);

void transpose_flt_thrcol(const float* restrict A, float* restrict B,
                          size_t A_rows, size_t A_cols,
                          size_t num_thr);
void transpose_dbl_thrcol(const double* restrict A, double* restrict B,
                          size_t A_rows, size_t A_cols,
                          size_t num_thr);
void transpose_fcmplx_thrcol(const float complex* restrict A,
                             float complex* restrict B,
                             size_t A_rows, size_t A_cols,
                             size_t num_thr);
void transpose_dcmplx_thrcol(const double complex* restrict A,
                             double complex* restrict B,
                             size_t A_rows, size_t A_cols,
                             size_t num_thr);

void transpose_flt_thrrow_blocked(const float* restrict A, float* restrict B,
                                  size_t A_rows, size_t A_cols,
                                  size_t num_thr,
                                  size_t blk_rows, size_t blk_cols);
void transpose_dbl_thrrow_blocked(const double* restrict A, double* restrict B,
                                  size_t A_rows, size_t A_cols,
                                  size_t num_thr,
                                  size_t blk_rows, size_t blk_cols);
void transpose_fcmplx_thrrow_blocked(const float complex* restrict A,
                                     float complex* restrict B,
                                     size_t A_rows, size_t A_cols,
                                     size_t num_thr,
                                     size_t blk_rows, size_t blk_cols);
void transpose_dcmplx_thrrow_blocked(const double complex* restrict A,
                                     double complex* restrict B,
                                     size_t A_rows, size_t A_cols,
                                     size_t num_thr,
                                     size_t blk_rows, size_t blk_cols);

void transpose_flt_thrcol_blocked(const float* restrict A, float* restrict B,
                                  size_t A_rows, size_t A_cols,
                                  size_t num_thr,
                                  size_t blk_rows, size_t blk_cols);
void transpose_dbl_thrcol_blocked(const double* restrict A, double* restrict B,
                                  size_t A_rows, size_t A_cols,
                                  size_t num_thr,
                                  size_t blk_rows, size_t blk_cols);
void transpose_fcmplx_thrcol_blocked(const float complex* restrict A,
                                     float complex* restrict B,
                                     size_t A_rows, size_t A_cols,
                                     size_t num_thr,
                                     size_t blk_rows, size_t blk_cols);
void transpose_dcmplx_thrcol_blocked(const double complex* restrict A,
                                     double complex* restrict B,
                                     size_t A_rows, size_t A_cols,
                                     size_t num_thr,
                                     size_t blk_rows, size_t blk_cols);

#endif /* TRANSPOSE_THREADS_H */
