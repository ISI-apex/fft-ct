/**
 * Transpose functions.
 *
 * @author Connor Imes <cimes@isi.edu>
 * @date 2019-07-15 
 */
#ifndef TRANSPOSE_FFTW_H
#define TRANSPOSE_FFTW_H

#include <complex.h>
#include <stdlib.h>

#include <fftw3.h>

void transpose_fftw_naive(const fftw_complex* restrict A,
                          fftw_complex* restrict B,
                          size_t A_rows, size_t A_cols);

void transpose_fftw_blocked(const fftw_complex* restrict A,
                            fftw_complex* restrict B,
                            size_t A_rows, size_t A_cols,
                            size_t blk_rows, size_t blk_cols);

#endif /* TRANSPOSE_FFTW_H */
