/**
 * Transpose functions.
 *
 * @author Kaushik Datta <kdatta@isi.edu>
 * @author Connor Imes <cimes@isi.edu>
 * @date 2019-08-07
 */
#ifndef TRANSPOSE_AVX_H
#define TRANSPOSE_AVX_H

#include <stdlib.h>

void transpose_dbl_avx_intr_8x8(const double* restrict A, double* restrict B,
                                size_t A_rows, size_t A_cols);

#endif /* TRANSPOSE_AVX_H */
