/**
 * Transpose functions.
 *
 * @author Connor Imes <cimes@isi.edu>
 * @date 2019-08-20
 */
#ifndef TRANSPOSE_FFTW_THREADS_H
#define TRANSPOSE_FFTW_THREADS_H

#include <complex.h>
#include <stdlib.h>

#include <fftw3.h>

void transpose_fftw_thrrow(const fftw_complex* restrict A,
                           fftw_complex* restrict B,
                           size_t A_rows, size_t A_cols,
                           size_t num_thr);

void transpose_fftw_thrcol(const fftw_complex* restrict A,
                           fftw_complex* restrict B,
                           size_t A_rows, size_t A_cols,
                           size_t num_thr);

void transpose_fftw_thrrow_blocked(const fftw_complex* restrict A,
                                   fftw_complex* restrict B,
                                   size_t A_rows, size_t A_cols,
                                   size_t num_thr,
                                   size_t blk_rows, size_t blk_cols);

void transpose_fftw_thrcol_blocked(const fftw_complex* restrict A,
                                   fftw_complex* restrict B,
                                   size_t A_rows, size_t A_cols,
                                   size_t num_thr,
                                   size_t blk_rows, size_t blk_cols);

#endif /* TRANSPOSE_FFTW_THREADS_H */
