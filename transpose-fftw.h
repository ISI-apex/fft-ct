/**
 * Transpose functions.
 *
 * @author Connor Imes <cimes@isi.edu>
 * @date 2019-07-15 
 */
#ifndef TRANSPOSE_FFTW_H
#define TRANSPOSE_FFTW_H

#include <stdlib.h>

#include <fftw3.h>

void transpose_fftw_complex_naive(fftw_complex* restrict A,
                                  fftw_complex* restrict B,
                                  size_t A_rows, size_t A_cols);

#endif /* TRANSPOSE_FFTW_H */