/**
 * Transpose functions.
 *
 * @author Connor Imes <cimes@isi.edu>
 * @author Kaushik Datta <kdatta@isi.edu>
 * @date 2019-07-15
 */
#include <complex.h>
#include <stdlib.h>

#include "transpose.h"

#define TRANSPOSE_BLK(A, B, A_rows, A_cols, r_min, c_min, r_max, c_max) { \
    size_t r, c; \
    for (r = (r_min); r < (r_max); r++) { \
        for (c = (c_min); c < (c_max); c++) { \
            (B)[(c) * (A_rows) + (r)] = (A)[(r) * (A_cols) + (c)]; \
        } \
    } \
}

#define TRANSPOSE_BLOCKED(A, B, A_rows, A_cols, blk_rows, blk_cols) { \
    /* take the ceiling of (A_rows / blk_rows) */ \
    const size_t n_rblks = (A_rows + blk_rows - 1) / blk_rows; \
    /* take the ceiling of (A_cols / blk_cols) */ \
    const size_t n_cblks = (A_cols + blk_cols - 1) / blk_cols; \
    const size_t n_full_rblks = A_rows / blk_rows; \
    const size_t n_full_cblks = A_cols / blk_cols; \
    const size_t rblk_remainder = A_rows % blk_rows; \
    const size_t cblk_remainder = A_cols % blk_cols; \
    size_t rblk_num, cblk_num, r_min, c_min, r_max, c_max; \
    /* perform transpose over all blocks (both full and partial) */ \
    for (rblk_num = 0; rblk_num < n_rblks; rblk_num++) { \
        r_min = rblk_num * blk_rows; \
        if (rblk_num < n_full_rblks) { \
            r_max = r_min + blk_rows; \
        } else { \
            r_max = r_min + rblk_remainder; \
        } \
        for (cblk_num = 0; cblk_num < n_cblks; cblk_num++) { \
            c_min = cblk_num * blk_cols; \
            if (cblk_num < n_full_cblks) { \
                c_max = c_min + blk_cols; \
            } else { \
                c_max = c_min + cblk_remainder; \
            } \
            /* perform actual transpose over current block */ \
            TRANSPOSE_BLK(A, B, A_rows, A_cols, r_min, c_min, r_max, c_max); \
        } \
    } \
}

void transpose_flt_naive(const float* restrict A, float* restrict B,
                         size_t A_rows, size_t A_cols)
{
    TRANSPOSE_BLK(A, B, A_rows, A_cols, 0, 0, A_rows, A_cols);
}

void transpose_dbl_naive(const double* restrict A, double* restrict B,
                         size_t A_rows, size_t A_cols)
{
    TRANSPOSE_BLK(A, B, A_rows, A_cols, 0, 0, A_rows, A_cols);
}

void transpose_fcmplx_naive(const float complex* restrict A,
                            float complex* restrict B,
                            size_t A_rows, size_t A_cols)
{
    TRANSPOSE_BLK(A, B, A_rows, A_cols, 0, 0, A_rows, A_cols);
}

void transpose_dcmplx_naive(const double complex* restrict A,
                            double complex* restrict B,
                            size_t A_rows, size_t A_cols)
{
    TRANSPOSE_BLK(A, B, A_rows, A_cols, 0, 0, A_rows, A_cols);
}

void transpose_flt_blocked(const float* restrict A, float* restrict B,
                           size_t A_rows, size_t A_cols,
                           size_t blk_rows, size_t blk_cols)
{
    TRANSPOSE_BLOCKED(A, B, A_rows, A_cols, blk_rows, blk_cols);
}

void transpose_dbl_blocked(const double* restrict A, double* restrict B,
                           size_t A_rows, size_t A_cols,
                           size_t blk_rows, size_t blk_cols)
{
    TRANSPOSE_BLOCKED(A, B, A_rows, A_cols, blk_rows, blk_cols);
}

void transpose_fcmplx_blocked(const float complex* restrict A,
                              float complex* restrict B,
                              size_t A_rows, size_t A_cols,
                              size_t blk_rows, size_t blk_cols)
{
    TRANSPOSE_BLOCKED(A, B, A_rows, A_cols, blk_rows, blk_cols);
}

void transpose_dcmplx_blocked(const double complex* restrict A,
                              double complex* restrict B,
                              size_t A_rows, size_t A_cols,
                              size_t blk_rows, size_t blk_cols)
{
    TRANSPOSE_BLOCKED(A, B, A_rows, A_cols, blk_rows, blk_cols);
}
