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

void transpose_fftw_complex_naive(const fftw_complex* restrict A,
                                  fftw_complex* restrict B,
                                  size_t A_rows, size_t A_cols)
{
    transpose_dbl_cmplx_naive(A, B, A_rows, A_cols);
}
