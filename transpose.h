/**
 * Transpose functions.
 *
 * @author Connor Imes <cimes@isi.edu>
 * @date 2019-07-15 
 */
#ifndef TRANSPOSE_H
#define TRANSPOSE_H

#include <stdlib.h>

void transpose_flt_naive(const float* restrict A, float* restrict B,
                         size_t A_rows, size_t A_cols);

void transpose_flt_blocked(const float* restrict A, float* restrict B,
                           size_t A_rows, size_t A_cols,
                           size_t blk_rows, size_t blk_cols);

void transpose_dbl_naive(const double* restrict A, double* restrict B,
                         size_t A_rows, size_t A_cols);

void transpose_dbl_blocked(const double* restrict A, double* restrict B,
                           size_t A_rows, size_t A_cols,
                           size_t blk_rows, size_t blk_cols);

#endif /* TRANSPOSE_H */
