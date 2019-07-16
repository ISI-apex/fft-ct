/**
 * Transpose functions
 *
 * @author Connor Imes <cimes@isi.edu>
 * @date 2019-07-15 
 */
#include <stdio.h>
#include <stdlib.h>

#include <fftw3.h>

#include "util.h"

void transpose(fftw_complex *A, fftw_complex *B,
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

#ifndef TEST_ROWS
#define TEST_ROWS 2
#endif

#ifndef TEST_COLS
#define TEST_COLS 3
#endif

static int check_transpose(fftw_complex A[TEST_ROWS][TEST_COLS],
                           fftw_complex B[TEST_COLS][TEST_ROWS])
{
    size_t r, c;
    for (r = 0; r < TEST_ROWS; r++)
        for (c = 0; c < TEST_COLS; c++)
            if (!is_complex_eq(A[r][c], B[c][r]))
                return -1;
    return 0;
}

int test_transpose(void)
{
    fftw_complex A[TEST_ROWS][TEST_COLS];
    fftw_complex B[TEST_COLS][TEST_ROWS];
    printf("Testing transpose of %ux%u matrix\n", TEST_ROWS, TEST_COLS);
    // init matrix
    complex_fill_rand(&A[0][0], TEST_ROWS * TEST_COLS);
    // execute
    printf("In:\n");
    matrix_print(&A[0][0], TEST_ROWS, TEST_COLS);
    transpose(&A[0][0], &B[0][0], TEST_ROWS, TEST_COLS);
    printf("Out:\n");
    matrix_print(&B[0][0], TEST_COLS, TEST_ROWS);
    // verify
    return check_transpose(A, B);
}
