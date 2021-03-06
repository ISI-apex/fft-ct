/**
 * Transpose functions.
 *
 * @author Connor Imes <cimes@isi.edu>
 * @date 2019-08-20
 */
#include <complex.h>
#include <stdlib.h>

#include <fftw3.h>

#include "transpose-fftw-threads.h"
#include "transpose-threads.h"

void transpose_fftw_thrrow(const fftw_complex* restrict A,
                           fftw_complex* restrict B,
                           size_t A_rows, size_t A_cols,
                           size_t num_thr)
{
    transpose_dcmplx_thrrow(A, B, A_rows, A_cols, num_thr);
}

void transpose_fftw_thrcol(const fftw_complex* restrict A,
                           fftw_complex* restrict B,
                           size_t A_rows, size_t A_cols,
                           size_t num_thr)
{
    transpose_dcmplx_thrcol(A, B, A_rows, A_cols, num_thr);

}

void transpose_fftw_thrrow_blocked(const fftw_complex* restrict A,
                                   fftw_complex* restrict B,
                                   size_t A_rows, size_t A_cols,
                                   size_t num_thr,
                                   size_t blk_rows, size_t blk_cols)
{
    transpose_dcmplx_thrrow_blocked(A, B, A_rows, A_cols, num_thr,
                                    blk_rows, blk_cols);
}

void transpose_fftw_thrcol_blocked(const fftw_complex* restrict A,
                                   fftw_complex* restrict B,
                                   size_t A_rows, size_t A_cols,
                                   size_t num_thr,
                                   size_t blk_rows, size_t blk_cols)
{
    transpose_dcmplx_thrcol_blocked(A, B, A_rows, A_cols, num_thr,
                                    blk_rows, blk_cols);
}
