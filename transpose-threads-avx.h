/**
 * Transpose functions.
 *
 * @author Kaushik Datta <kdatta@isi.edu>
 * @date 2019-08-15
 */
#ifndef TRANSPOSE_THREADS_AVX_H
#define TRANSPOSE_THREADS_AVX_H

#include <stdlib.h>

void transpose_dbl_thrrow_avx512_intr(const double* restrict A,
                                      double* restrict B,
                                      size_t A_rows, size_t A_cols,
                                      size_t num_thr);

void transpose_dbl_thrcol_avx512_intr(const double* restrict A,
                                      double* restrict B,
                                      size_t A_rows, size_t A_cols,
                                      size_t num_thr);

#endif /* TRANSPOSE_THREADS_AVX_H */
