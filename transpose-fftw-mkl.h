/**
 * Transpose functions.
 *
 * @author Connor Imes <cimes@isi.edu>
 * @date 2019-09-13
 */
#ifndef TRANSPOSE_FFTW_MKL_H
#define TRANSPOSE_FFTW_MKL_H

#include <complex.h>
#include <stdlib.h>

#include <fftw3.h>

void transpose_fftw_mkl(const fftw_complex* restrict A,
                        fftw_complex* restrict B,
                        size_t A_rows, size_t A_cols);

#endif /* TRANSPOSE_FFTW_MKL_H */
