/**
 * Transpose functions.
 *
 * @author Connor Imes <cimes@isi.edu>
 * @date 2019-08-20
 */
#ifndef TRANSPOSE_FFTWF_THREADS_AVX
#define TRANSPOSE_FFTWF_THREADS_AVX

#include <complex.h>
#include <stdlib.h>

#include <fftw3.h>

void transpose_fftwf_thrrow_avx512_intr(const fftwf_complex* restrict A,
                                        fftwf_complex* restrict B,
                                        size_t A_rows, size_t A_cols,
                                        size_t num_thr);

void transpose_fftwf_thrcol_avx512_intr(const fftwf_complex* restrict A,
                                        fftwf_complex* restrict B,
                                        size_t A_rows, size_t A_cols,
                                        size_t num_thr);

#endif /* TRANSPOSE_FFTWF_THREADS_AVX */
