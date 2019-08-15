/**
 * Transpose functions.
 *
 * @author Kaushik Datta <kdatta@isi.edu>
 * @date 2019-08-15
 */
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>

// intrinsics
#include <immintrin.h>

#include "transpose-threads-avx.h"
#include "util.h"

void transpose_dbl_threads_avx_intr_8x8_row(const double* restrict A,
                                            double* restrict B,
                                            size_t A_rows, size_t A_cols,
                                            size_t num_thr)
{
    // TODO
    exit(-1);
}

void transpose_dbl_threads_avx_intr_8x8_col(const double* restrict A,
                                            double* restrict B,
                                            size_t A_rows, size_t A_cols,
                                            size_t num_thr)
{
    // TODO
    exit(-1);
}
