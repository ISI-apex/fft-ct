/**
 * Transpose functions.
 *
 * @author Connor Imes <cimes@isi.edu>
 * @date 2019-07-15 
 */
#ifndef TRANSPOSE_FFTWF_H
#define TRANSPOSE_FFTWF_H

#include <complex.h>
#include <stdlib.h>

#include <fftw3.h>

void transpose_fftwf_naive(const fftwf_complex* restrict A,
                           fftwf_complex* restrict B,
                           size_t A_rows, size_t A_cols);

void transpose_fftwf_blocked(const fftwf_complex* restrict A,
                             fftwf_complex* restrict B,
                             size_t A_rows, size_t A_cols,
                             size_t blk_rows, size_t blk_cols);

#endif /* TRANSPOSE_FFTWF_H */
