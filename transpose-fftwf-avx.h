/**
 * Transpose functions.
 *
 * @author Connor Imes <cimes@isi.edu>
 * @date 2019-08-20
 */
#ifndef TRANSPOSE_FFTWF_AVX_H
#define TRANSPOSE_FFTWF_AVX_H

#include <complex.h>
#include <stdlib.h>

#include <fftw3.h>

void transpose_fftwf_avx512_intr(const fftwf_complex* restrict A,
                                 fftwf_complex* restrict B,
                                 size_t A_rows, size_t A_cols);

#endif /* TRANSPOSE_FFTWF_AVX_H */
