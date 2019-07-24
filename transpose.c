/**
 * Transpose functions.
 *
 * @author Connor Imes <cimes@isi.edu>
 * @date 2019-07-15 
 */
#include <stdlib.h>

#include <fftw3.h>

#include "transpose.h"

void transpose(fftw_complex *A, fftw_complex *B, size_t A_rows, size_t A_cols)
{
    size_t r, c;
    for (r = 0; r < A_rows; r++) {
        for (c = 0; c < A_cols; c++) {
            B[c * A_rows + r][0] = A[r * A_cols + c][0]; // re
            B[c * A_rows + r][1] = A[r * A_cols + c][1]; // im
        }
    }
}
