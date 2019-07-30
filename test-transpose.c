/**
 * Transpose test.
 *
 * @author Connor Imes <cimes@isi.edu>
 * @date 2019-07-15 
 */
#include <stdio.h>
#include <stdlib.h>

#include <fftw3.h>

#include <mkl.h>

#include "transpose.h"
#include "util.h"

#ifndef TEST_ROWS
#define TEST_ROWS 2
#endif

#ifndef TEST_COLS
#define TEST_COLS 3
#endif

static int check_transpose_fftw_complex(fftw_complex A[TEST_ROWS][TEST_COLS],
                                        fftw_complex B[TEST_COLS][TEST_ROWS])
{
    size_t r, c;
    for (r = 0; r < TEST_ROWS; r++)
        for (c = 0; c < TEST_COLS; c++)
            if (!is_eq_fftw_complex(A[r][c], B[c][r]))
                return -1;
    return 0;
}

static int check_transpose_dbl(double A[TEST_ROWS][TEST_COLS],
                               double B[TEST_COLS][TEST_ROWS])
{
    size_t r, c;
    for (r = 0; r < TEST_ROWS; r++)
        for (c = 0; c < TEST_COLS; c++)
            if (!is_eq_dbl(A[r][c], B[c][r]))
                return -1;
    return 0;
}

static int check_transpose_cmplx16(MKL_Complex16 A[TEST_ROWS][TEST_COLS],
                                   MKL_Complex16 B[TEST_COLS][TEST_ROWS])
{
    size_t r, c;
    for (r = 0; r < TEST_ROWS; r++)
        for (c = 0; c < TEST_COLS; c++)
            if (!is_eq_cmplx16(A[r][c], B[c][r]))
                return -1;
    return 0;
}

static int test_transpose_fftw_complex_naive(void)
{
    fftw_complex A[TEST_ROWS][TEST_COLS];
    fftw_complex B[TEST_COLS][TEST_ROWS];
    printf("Testing transpose of %ux%u matrix\n", TEST_ROWS, TEST_COLS);
    // init matrix
    fill_rand_fftw_complex(&A[0][0], TEST_ROWS * TEST_COLS);
    // execute
    printf("In:\n");
    matrix_print_fftw_complex(&A[0][0], TEST_ROWS, TEST_COLS);
    transpose_fftw_complex_naive(&A[0][0], &B[0][0], TEST_ROWS, TEST_COLS);
    printf("Out:\n");
    matrix_print_fftw_complex(&B[0][0], TEST_COLS, TEST_ROWS);
    // verify
    return check_transpose_fftw_complex(A, B);
}

static int test_transpose_dbl_mkl(void)
{
    double A[TEST_ROWS][TEST_COLS];
    double B[TEST_COLS][TEST_ROWS];
    printf("Testing transpose of %ux%u matrix\n", TEST_ROWS, TEST_COLS);
    // init matrix
    fill_rand_dbl(&A[0][0], TEST_ROWS * TEST_COLS);
    // execute
    printf("In:\n");
    matrix_print_dbl(&A[0][0], TEST_ROWS, TEST_COLS);
    transpose_dbl_mkl(&A[0][0], &B[0][0], TEST_ROWS, TEST_COLS);
    printf("Out:\n");
    matrix_print_dbl(&B[0][0], TEST_COLS, TEST_ROWS);
    // verify
    return check_transpose_dbl(A, B);
}

static int test_transpose_cmplx16_mkl(void)
{
    MKL_Complex16 A[TEST_ROWS][TEST_COLS];
    MKL_Complex16 B[TEST_COLS][TEST_ROWS];
    printf("Testing transpose of %ux%u matrix\n", TEST_ROWS, TEST_COLS);
    // init matrix
    fill_rand_cmplx16(&A[0][0], TEST_ROWS * TEST_COLS);
    // execute
    printf("In:\n");
    matrix_print_cmplx16(&A[0][0], TEST_ROWS, TEST_COLS);
    transpose_cmplx16_mkl(&A[0][0], &B[0][0], TEST_ROWS, TEST_COLS);
    printf("Out:\n");
    matrix_print_cmplx16(&B[0][0], TEST_COLS, TEST_ROWS);
    // verify
    return check_transpose_cmplx16(A, B);
}

int main(void)
{
    int ret = 0;
    int rc;
    printf("transpose_fftw_complex:\n");
    rc = test_transpose_fftw_complex_naive();
    ret |= rc;
    printf("%s\n", rc ? "Failed" : "Success");
    printf("transpose_dbl_mkl:\n");
    rc = test_transpose_dbl_mkl();
    ret |= rc;
    printf("%s\n", rc ? "Failed" : "Success");
    printf("transpose_cmplx16_mkl:\n");
    rc = test_transpose_cmplx16_mkl();
    ret |= rc;
    printf("%s\n", rc ? "Failed" : "Success");
    return ret;
}
