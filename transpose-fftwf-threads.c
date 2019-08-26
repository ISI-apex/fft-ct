/**
 * Transpose functions.
 *
 * @author Connor Imes <cimes@isi.edu>
 * @date 2019-08-20
 */
#include <complex.h>
#include <stdlib.h>

#include <fftw3.h>

#include "transpose-fftwf-threads.h"
#include "transpose-threads.h"

void transpose_fftwf_threads_row(const fftwf_complex* restrict A,
                                 fftwf_complex* restrict B,
                                 size_t A_rows, size_t A_cols,
                                 size_t num_thr)
{
    transpose_fcmplx_threads_row(A, B, A_rows, A_cols, num_thr);
}

void transpose_fftwf_threads_col(const fftwf_complex* restrict A,
                                 fftwf_complex* restrict B,
                                 size_t A_rows, size_t A_cols,
                                 size_t num_thr)
{
    transpose_fcmplx_threads_col(A, B, A_rows, A_cols, num_thr);

}

void transpose_fftwf_threads_row_blocked(const fftwf_complex* restrict A,
                                         fftwf_complex* restrict B,
                                         size_t A_rows, size_t A_cols,
                                         size_t num_thr,
                                         size_t blk_rows, size_t blk_cols)
{
    transpose_fcmplx_threads_row_blocked(A, B, A_rows, A_cols, num_thr,
                                         blk_rows, blk_cols);
}

void transpose_fftwf_threads_col_blocked(const fftwf_complex* restrict A,
                                         fftwf_complex* restrict B,
                                         size_t A_rows, size_t A_cols,
                                         size_t num_thr,
                                         size_t blk_rows, size_t blk_cols)
{
    transpose_fcmplx_threads_col_blocked(A, B, A_rows, A_cols, num_thr,
                                         blk_rows, blk_cols);
}