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
#include "transpose-fftw.h"

void transpose_fftw_naive(const fftw_complex* restrict A,
                          fftw_complex* restrict B,
                          size_t A_rows, size_t A_cols)
{
    transpose_dcmplx_naive(A, B, A_rows, A_cols);
}

void transpose_fftw_blocked(const fftw_complex* restrict A,
                            fftw_complex* restrict B,
                            size_t A_rows, size_t A_cols,
                            size_t blk_rows, size_t blk_cols)
{
    transpose_dcmplx_blocked(A, B, A_rows, A_cols, blk_rows, blk_cols);
}
