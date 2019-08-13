/**
 * Transpose functions.
 *
 * @author Connor Imes <cimes@isi.edu>
 * @date 2019-07-15 
 */
#include <stdlib.h>

#include <fftw3.h>

#include "transpose-fftw.h"

#define TRANSPOSE_CMPLX_BLK(A, B, A_rows, A_cols, r_min, c_min, r_max, c_max) { \
    size_t r, c; \
    for (r = (r_min); r < (r_max); r++) { \
        for (c = (c_min); c < (c_max); c++) { \
            (B)[(c) * (A_rows) + (r)][0] = (A)[(r) * (A_cols) + (c)][0]; /* re */ \
            (B)[(c) * (A_rows) + (r)][1] = (A)[(r) * (A_cols) + (c)][1]; /* im */ \
        } \
    } \
}

void transpose_fftwf_complex_naive(fftwf_complex* restrict A,
                                   fftwf_complex* restrict B,
                                   size_t A_rows, size_t A_cols)
{
    TRANSPOSE_CMPLX_BLK(A, B, A_rows, A_cols, 0, 0, A_rows, A_cols);
}

void transpose_fftw_complex_naive(fftw_complex* restrict A,
                                  fftw_complex* restrict B,
                                  size_t A_rows, size_t A_cols)
{
    TRANSPOSE_CMPLX_BLK(A, B, A_rows, A_cols, 0, 0, A_rows, A_cols);
}
