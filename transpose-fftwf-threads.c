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

void transpose_fftwf_thrrow(const fftwf_complex* restrict A,
                            fftwf_complex* restrict B,
                            size_t A_rows, size_t A_cols,
                            size_t num_thr)
{
    transpose_fcmplx_thrrow(A, B, A_rows, A_cols, num_thr);
}

void transpose_fftwf_thrcol(const fftwf_complex* restrict A,
                            fftwf_complex* restrict B,
                            size_t A_rows, size_t A_cols,
                            size_t num_thr)
{
    transpose_fcmplx_thrcol(A, B, A_rows, A_cols, num_thr);

}

void transpose_fftwf_thrrow_blocked(const fftwf_complex* restrict A,
                                    fftwf_complex* restrict B,
                                    size_t A_rows, size_t A_cols,
                                    size_t num_thr,
                                    size_t blk_rows, size_t blk_cols)
{
    transpose_fcmplx_thrrow_blocked(A, B, A_rows, A_cols, num_thr,
                                    blk_rows, blk_cols);
}

void transpose_fftwf_thrcol_blocked(const fftwf_complex* restrict A,
                                    fftwf_complex* restrict B,
                                    size_t A_rows, size_t A_cols,
                                    size_t num_thr,
                                    size_t blk_rows, size_t blk_cols)
{
    transpose_fcmplx_thrcol_blocked(A, B, A_rows, A_cols, num_thr,
                                    blk_rows, blk_cols);
}
