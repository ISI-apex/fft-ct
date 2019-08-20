/**
 * Transpose functions.
 *
 * @author Connor Imes <cimes@isi.edu>
 * @date 2019-08-20
 */
#ifndef TRANSPOSE_THREADS_FFTW_H
#define TRANSPOSE_THREADS_FFTW_H

#include <complex.h>
#include <stdlib.h>

#include <fftw3.h>

void transpose_fftw_complex_threads_row(const fftw_complex* restrict A,
                                        fftw_complex* restrict B,
                                        size_t A_rows, size_t A_cols,
                                        size_t num_thr);

void transpose_fftw_complex_threads_col(const fftw_complex* restrict A,
                                        fftw_complex* restrict B,
                                        size_t A_rows, size_t A_cols,
                                        size_t num_thr);

void transpose_fftw_complex_threads_row_blocked(const fftw_complex* restrict A,
                                                fftw_complex* restrict B,
                                                size_t A_rows, size_t A_cols,
                                                size_t num_thr,
                                                size_t blk_rows,
                                                size_t blk_cols);

void transpose_fftw_complex_threads_col_blocked(const fftw_complex* restrict A,
                                                fftw_complex* restrict B,
                                                size_t A_rows, size_t A_cols,
                                                size_t num_thr,
                                                size_t blk_rows,
                                                size_t blk_cols);

#endif /* TRANSPOSE_THREADS_FFTW_H */
