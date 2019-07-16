/**
 * Transpose functions.
 *
 * @author Connor Imes <cimes@isi.edu>
 * @date 2019-07-15 
 */
#ifndef TRANSPOSE_H
#define TRANSPOSE_H

#include <stdlib.h>

#include <fftw3.h>

void transpose(const fftw_complex *A, fftw_complex *B,
               size_t A_rows, size_t A_cols);

int test_transpose(void);

#endif /* TRANSPOSE_H */
