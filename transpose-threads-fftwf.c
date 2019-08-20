/**
 * Transpose functions.
 *
 * @author Connor Imes <cimes@isi.edu>
 * @date 2019-07-15 
 */
#include <complex.h>
#include <stdlib.h>

#include <fftw3.h>

#include "transpose-threads.h"
#include "transpose-threads-fftwf.h"

void transpose_fftwf_complex_threads_row(const fftwf_complex* restrict A,
                                         fftwf_complex* restrict B,
                                         size_t A_rows, size_t A_cols,
                                         size_t num_thr)
{
    transpose_flt_cmplx_threads_row(A, B, A_rows, A_cols, num_thr);
}

void transpose_fftwf_complex_threads_col(const fftwf_complex* restrict A,
                                         fftwf_complex* restrict B,
                                         size_t A_rows, size_t A_cols,
                                         size_t num_thr)
{
    transpose_flt_cmplx_threads_col(A, B, A_rows, A_cols, num_thr);

}

void transpose_fftwf_complex_threads_row_blocked(const fftwf_complex* restrict A,
                                                 fftwf_complex* restrict B,
                                                 size_t A_rows, size_t A_cols,
                                                 size_t num_thr,
                                                 size_t blk_rows,
                                                 size_t blk_cols)
{
    transpose_flt_cmplx_threads_row_blocked(A, B, A_rows, A_cols, num_thr,
                                            blk_rows, blk_cols);
}

void transpose_fftwf_complex_threads_col_blocked(const fftwf_complex* restrict A,
                                                 fftwf_complex* restrict B,
                                                 size_t A_rows, size_t A_cols,
                                                 size_t num_thr,
                                                 size_t blk_rows,
                                                 size_t blk_cols)
{
    transpose_flt_cmplx_threads_col_blocked(A, B, A_rows, A_cols, num_thr,
                                            blk_rows, blk_cols);
}
