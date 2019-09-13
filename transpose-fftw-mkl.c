/**
 * Transpose functions.
 *
 * @author Connor Imes <cimes@isi.edu>
 * @date 2019-09-13
 */
#include <complex.h>
#include <stdlib.h>

#include <fftw3.h>

#include "transpose-fftw-mkl.h"
#include "transpose-mkl.h"

void transpose_fftw_mkl(const fftw_complex* restrict A,
                        fftw_complex* restrict B,
                        size_t A_rows, size_t A_cols)
{
    transpose_cmplx16_mkl(A, B, A_rows, A_cols);
}
