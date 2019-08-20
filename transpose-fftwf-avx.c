/**
 * Transpose functions.
 *
 * @author Connor Imes <cimes@isi.edu>
 * @date 2019-08-20
 */
#include <complex.h>
#include <stdlib.h>

#include <fftw3.h>

#include "transpose-avx.h"
#include "transpose-fftwf-avx.h"

void transpose_fftwf_avx512_intr(const fftwf_complex* restrict A,
                                 fftwf_complex* restrict B,
                                 size_t A_rows, size_t A_cols)
{
    transpose_dbl_avx512_intr((const double* restrict)A, (double* restrict)B,
                              A_rows, A_cols);
}
