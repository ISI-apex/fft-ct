/**
 * Transpose functions.
 *
 * @author Connor Imes <cimes@isi.edu>
 * @date 2019-07-15 
 */
#include <stdlib.h>

#include <fftw3.h>

#include <mkl.h>

#include "transpose.h"

void transpose_flt_naive(const float* restrict A, float* restrict B,
                         size_t A_rows, size_t A_cols) {
    size_t r, c;
    for (r = 0; r < A_rows; r++) {
        for (c = 0; c < A_cols; c++) {
            B[c * A_rows + r] = A[r * A_cols + c];
        }
    }
}

void transpose_flt_blocked(const float* restrict A, float* restrict B,
                           size_t A_rows, size_t A_cols,
                           size_t blk_rows, size_t blk_cols) {
    size_t r, c, r_min, c_min, r_max, c_max;
    size_t num_row_blocks, num_col_blocks;
    size_t num_full_row_blocks, num_full_col_blocks;
    size_t row_block_remainder, col_block_remainder;
    size_t row_block_num, col_block_num;

    // take the ceiling of (A_rows / blk_rows)
    num_row_blocks = (A_rows + blk_rows - 1) / blk_rows;
    // take the ceiling of (A_cols / blk_cols)
    num_col_blocks = (A_cols + blk_cols - 1) / blk_cols;

    num_full_row_blocks = A_rows / blk_rows;
    num_full_col_blocks = A_cols / blk_cols;

    row_block_remainder = A_rows % blk_rows;
    col_block_remainder = A_cols % blk_cols;

    // perform transpose over all blocks (both full and partial)
    for (row_block_num = 0; row_block_num < num_row_blocks; row_block_num++) {
        for (col_block_num = 0; col_block_num < num_col_blocks; col_block_num++) {
            r_min = row_block_num * blk_rows;
            c_min = col_block_num * blk_cols;

            if (row_block_num < num_full_row_blocks) {
                r_max = r_min + blk_rows;
            } else {
                r_max = r_min + row_block_remainder;
            }

            if (col_block_num < num_full_col_blocks) {
                c_max = c_min + blk_cols;
            } else {
                c_max = c_min + col_block_remainder;
            }

            // perform actual transpose over current block
            for (r = r_min; r < r_max; r++) {
                for (c = c_min; c < c_max; c++) {
                    B[c * A_rows + r] = A[r * A_cols + c];
                }
            }
        }
    }
}

void transpose_dbl_naive(const double* restrict A, double* restrict B,
                         size_t A_rows, size_t A_cols) {
    size_t r, c;
    for (r = 0; r < A_rows; r++) {
        for (c = 0; c < A_cols; c++) {
            B[c * A_rows + r] = A[r * A_cols + c];
        }
    }
}

void transpose_dbl_blocked(const double* restrict A, double* restrict B,
                           size_t A_rows, size_t A_cols,
                           size_t blk_rows, size_t blk_cols) {
    size_t r, c, r_min, c_min, r_max, c_max;
    size_t num_row_blocks, num_col_blocks;
    size_t num_full_row_blocks, num_full_col_blocks;
    size_t row_block_remainder, col_block_remainder;
    size_t row_block_num, col_block_num;

    // take the ceiling of (A_rows / blk_rows)
    num_row_blocks = (A_rows + blk_rows - 1) / blk_rows;
    // take the ceiling of (A_cols / blk_cols)
    num_col_blocks = (A_cols + blk_cols - 1) / blk_cols;

    num_full_row_blocks = A_rows / blk_rows;
    num_full_col_blocks = A_cols / blk_cols;

    row_block_remainder = A_rows % blk_rows;
    col_block_remainder = A_cols % blk_cols;

    // perform transpose over all blocks (both full and partial)
    for (row_block_num = 0; row_block_num < num_row_blocks; row_block_num++) {
        for (col_block_num = 0; col_block_num < num_col_blocks; col_block_num++) {
            r_min = row_block_num * blk_rows;
            c_min = col_block_num * blk_cols;

            if (row_block_num < num_full_row_blocks) {
                r_max = r_min + blk_rows;
            } else {
                r_max = r_min + row_block_remainder;
            }

            if (col_block_num < num_full_col_blocks) {
                c_max = c_min + blk_cols;
            } else {
                c_max = c_min + col_block_remainder;
            }

            // perform actual transpose over current block
            for (r = r_min; r < r_max; r++) {
                for (c = c_min; c < c_max; c++) {
                    B[c * A_rows + r] = A[r * A_cols + c];
                }
            }
        }
    }
}

void transpose_fftw_complex_naive(fftw_complex* restrict A, fftw_complex* restrict B,
                                  size_t A_rows, size_t A_cols)
{
    size_t r, c;
    for (r = 0; r < A_rows; r++) {
        for (c = 0; c < A_cols; c++) {
            B[c * A_rows + r][0] = A[r * A_cols + c][0]; // re
            B[c * A_rows + r][1] = A[r * A_cols + c][1]; // im
        }
    }
}

void transpose_dbl_mkl(const double* restrict A, double* restrict B,
                       size_t A_rows, size_t A_cols)
{
    mkl_domatcopy('r', 't', A_rows, A_cols, 1, A, A_cols, B, A_rows);
}

void transpose_cmplx16_mkl(const MKL_Complex16* restrict A, MKL_Complex16* restrict B,
                           size_t A_rows, size_t A_cols)
{
    static const MKL_Complex16 alpha = {
        .real = 1,
        .imag = 0,
    };
    mkl_zomatcopy('r', 't', A_rows, A_cols, alpha, A, A_cols, B, A_rows);
}
