/**
 * Transpose functions.
 *
 * @author Connor Imes <cimes@isi.edu>
 * @date 2019-07-15 
 */
#include <complex.h>
#include <stdlib.h>

#include <fftw3.h>

#include "transpose.h"
#include "transpose-fftwf.h"

void transpose_fftwf_complex_naive(const fftwf_complex* restrict A,
                                   fftwf_complex* restrict B,
                                   size_t A_rows, size_t A_cols)
{
    transpose_flt_cmplx_naive(A, B, A_rows, A_cols);
}

void transpose_fftwf_complex_blocked(const fftwf_complex* restrict A,
                                     fftwf_complex* restrict B,
                                     size_t A_rows, size_t A_cols,
                                     size_t blk_rows, size_t blk_cols)
{
    transpose_flt_cmplx_blocked(A, B, A_rows, A_cols, blk_rows, blk_cols);
}
