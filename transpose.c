/**
 * Transpose functions.
 *
 * @author Connor Imes <cimes@isi.edu>
 * @date 2019-07-15 
 */
#include <stdlib.h>

#include <fftw3.h>

#include <mkl.h>

#include "transpose.h"

void transpose_flt_naive(const float* restrict A, float* restrict B,
			 size_t A_rows, size_t A_cols) {
    size_t r, c;
    for (r = 0; r < A_rows; r++) {
        for (c = 0; c < A_cols; c++) {
            B[c * A_rows + r] = A[r * A_cols + c];
        }
    }
}

void transpose_dbl_naive(const double* restrict A, double* restrict B,
			 size_t A_rows, size_t A_cols) {
    size_t r, c;
    for (r = 0; r < A_rows; r++) {
        for (c = 0; c < A_cols; c++) {
            B[c * A_rows + r] = A[r * A_cols + c];
        }
    }
}

void transpose_fftw_complex_naive(fftw_complex* restrict A, fftw_complex* restrict B,
                                  size_t A_rows, size_t A_cols)
{
    size_t r, c;
    for (r = 0; r < A_rows; r++) {
        for (c = 0; c < A_cols; c++) {
            B[c * A_rows + r][0] = A[r * A_cols + c][0]; // re
            B[c * A_rows + r][1] = A[r * A_cols + c][1]; // im
        }
    }
}

void transpose_dbl_mkl(const double* restrict A, double* restrict B,
                       size_t A_rows, size_t A_cols)
{
    mkl_domatcopy('r', 't', A_rows, A_cols, 1, A, A_cols, B, A_rows);
}

void transpose_cmplx16_mkl(const MKL_Complex16* restrict A, MKL_Complex16* restrict B,
                           size_t A_rows, size_t A_cols)
{
    static const MKL_Complex16 alpha = {
        .real = 1,
        .imag = 0,
    };
    mkl_zomatcopy('r', 't', A_rows, A_cols, alpha, A, A_cols, B, A_rows);
}
