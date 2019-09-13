/**
 * Transpose functions.
 *
 * @author Connor Imes <cimes@isi.edu>
 * @date 2019-09-13
 */
#ifndef TRANSPOSE_FFTWF_MKL_H
#define TRANSPOSE_FFTWF_MKL_H

#include <complex.h>
#include <stdlib.h>

#include <fftw3.h>

void transpose_fftwf_mkl(const fftwf_complex* restrict A,
                         fftwf_complex* restrict B,
                         size_t A_rows, size_t A_cols);

#endif /* TRANSPOSE_FFTWF_MKL_H */
