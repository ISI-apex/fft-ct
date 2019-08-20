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
#include "transpose-threads-fftw.h"

void transpose_fftw_complex_threads_row(const fftw_complex* restrict A,
                                        fftw_complex* restrict B,
                                        size_t A_rows, size_t A_cols,
                                        size_t num_thr)
{
    transpose_dbl_cmplx_threads_row(A, B, A_rows, A_cols, num_thr);
}

void transpose_fftw_complex_threads_col(const fftw_complex* restrict A,
                                        fftw_complex* restrict B,
                                        size_t A_rows, size_t A_cols,
                                        size_t num_thr)
{
    transpose_dbl_cmplx_threads_col(A, B, A_rows, A_cols, num_thr);

}

void transpose_fftw_complex_threads_row_blocked(const fftw_complex* restrict A,
                                                fftw_complex* restrict B,
                                                size_t A_rows, size_t A_cols,
                                                size_t num_thr,
                                                size_t blk_rows,
                                                size_t blk_cols)
{
    transpose_dbl_cmplx_threads_row_blocked(A, B, A_rows, A_cols, num_thr,
                                            blk_rows, blk_cols);
}

void transpose_fftw_complex_threads_col_blocked(const fftw_complex* restrict A,
                                                fftw_complex* restrict B,
                                                size_t A_rows, size_t A_cols,
                                                size_t num_thr,
                                                size_t blk_rows,
                                                size_t blk_cols)
{
    transpose_dbl_cmplx_threads_col_blocked(A, B, A_rows, A_cols, num_thr,
                                            blk_rows, blk_cols);
}
