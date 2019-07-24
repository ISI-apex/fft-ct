/**
 * Transpose test.
 *
 * @author Connor Imes <cimes@isi.edu>
 * @date 2019-07-15 
 */
#include <stdio.h>
#include <stdlib.h>

#include "transpose.h"
#include "util.h"

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

static int test_transpose(void)
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

int main(void)
{
    int rc = test_transpose();
    printf("%s\n", rc ? "Failed" : "Success");
    return rc;
}
