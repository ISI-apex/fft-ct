/**
 * Transpose functions.
 *
 * @author Connor Imes <cimes@isi.edu>
 * @date 2019-08-20
 */
#include <complex.h>
#include <stdlib.h>

#include <fftw3.h>

#include "transpose-fftwf-threads-avx.h"
#include "transpose-threads-avx.h"

void transpose_fftwf_thrrow_avx512_intr(const fftwf_complex* restrict A,
                                        fftwf_complex* restrict B,
                                        size_t A_rows, size_t A_cols,
                                        size_t num_thr)
{
    transpose_dbl_thrrow_avx512_intr((const double* restrict)A,
                                     (double* restrict)B,
                                     A_rows, A_cols, num_thr);
}

void transpose_fftwf_thrcol_avx512_intr(const fftwf_complex* restrict A,
                                        fftwf_complex* restrict B,
                                        size_t A_rows, size_t A_cols,
                                        size_t num_thr)
{
    transpose_dbl_thrcol_avx512_intr((const double* restrict)A,
                                     (double* restrict)B,
                                     A_rows, A_cols, num_thr);
}
