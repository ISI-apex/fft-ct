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
#include "transpose-fftw.h"
#include "transpose-mkl.h"
#include "util.h"
#include "util-fftw.h"
#include "util-mkl.h"

#ifndef TEST_ROWS
#define TEST_ROWS 3
#endif

#ifndef TEST_COLS
#define TEST_COLS 5
#endif

#ifndef TEST_BLK_ROWS
#define TEST_BLK_ROWS 2
#endif

#ifndef TEST_BLK_COLS
#define TEST_BLK_COLS 3
#endif

#define num2str(x) str(x)
#define str(x) #x

#define CHECK_TRANSPOSE(A, B, fn_is_eq, rc) \
    size_t r, c; \
    rc = 0; \
    for (r = 0; r < TEST_ROWS && !rc; r++) { \
        for (c = 0; c < TEST_COLS && !rc; c++) { \
            rc = !fn_is_eq(A[r][c], B[c][r]); \
        } \
    }

#define TEST_TRANSPOSE(datatype, fn_fill, fn_mat_print, fn_transpose, fn_is_eq) \
    datatype A[TEST_ROWS][TEST_COLS]; \
    datatype B[TEST_COLS][TEST_ROWS]; \
    int rc; \
    fn_fill(&A[0][0], TEST_ROWS * TEST_COLS); \
    printf("In:\n"); \
    fn_mat_print(&A[0][0], TEST_ROWS, TEST_COLS); \
    fn_transpose(&A[0][0], &B[0][0], TEST_ROWS, TEST_COLS); \
    printf("Out:\n"); \
    fn_mat_print(&B[0][0], TEST_COLS, TEST_ROWS); \
    CHECK_TRANSPOSE(A, B, fn_is_eq, rc); \
    return rc;

#define TEST_TRANSPOSE_BLOCKED(datatype, fn_fill, fn_mat_print, fn_transpose, fn_is_eq) \
    datatype A[TEST_ROWS][TEST_COLS]; \
    datatype B[TEST_COLS][TEST_ROWS]; \
    int rc; \
    fn_fill(&A[0][0], TEST_ROWS * TEST_COLS); \
    printf("In:\n"); \
    fn_mat_print(&A[0][0], TEST_ROWS, TEST_COLS); \
    fn_transpose(&A[0][0], &B[0][0], TEST_ROWS, TEST_COLS, TEST_BLK_ROWS, TEST_BLK_COLS); \
    printf("Out:\n"); \
    fn_mat_print(&B[0][0], TEST_COLS, TEST_ROWS); \
    CHECK_TRANSPOSE(A, B, fn_is_eq, rc); \
    return rc;

static int test_transpose_flt_naive(void)
{
    TEST_TRANSPOSE(float, fill_rand_flt, matrix_print_flt, transpose_flt_naive,
                   is_eq_flt);
}

static int test_transpose_flt_blocked(void)
{
    TEST_TRANSPOSE_BLOCKED(float, fill_rand_flt, matrix_print_flt,
                           transpose_flt_blocked, is_eq_flt);
}

static int test_transpose_dbl_naive(void)
{
    TEST_TRANSPOSE(double, fill_rand_dbl, matrix_print_dbl, transpose_dbl_naive,
                   is_eq_dbl);
}

static int test_transpose_dbl_blocked(void)
{
    TEST_TRANSPOSE_BLOCKED(double, fill_rand_dbl, matrix_print_dbl,
                           transpose_dbl_blocked, is_eq_dbl);
}

static int test_transpose_fftw_complex_naive(void)
{
    TEST_TRANSPOSE(fftw_complex, fill_rand_fftw_complex,
                   matrix_print_fftw_complex, transpose_fftw_complex_naive,
                   is_eq_fftw_complex);
}

static int test_transpose_flt_mkl(void)
{
    TEST_TRANSPOSE(float, fill_rand_flt, matrix_print_flt, transpose_flt_mkl,
                   is_eq_flt);
}

static int test_transpose_dbl_mkl(void)
{
    TEST_TRANSPOSE(double, fill_rand_dbl, matrix_print_dbl, transpose_dbl_mkl,
                   is_eq_dbl);
}

static int test_transpose_cmplx8_mkl(void)
{
    TEST_TRANSPOSE(MKL_Complex8, fill_rand_cmplx8, matrix_print_cmplx8,
                   transpose_cmplx8_mkl, is_eq_cmplx8);
}

static int test_transpose_cmplx16_mkl(void)
{
    TEST_TRANSPOSE(MKL_Complex16, fill_rand_cmplx16, matrix_print_cmplx16,
                   transpose_cmplx16_mkl, is_eq_cmplx16);
}

int main(void)
{
    int ret = 0;
    int rc;

    printf("transpose_flt_naive:\n");
    rc = test_transpose_flt_naive();
    ret |= rc;
    printf("%s\n", rc ? "Failed" : "Success");

    printf("\ntranspose_flt_blocked (block size = %zux%zu):\n",
           (size_t) TEST_BLK_ROWS, (size_t) TEST_BLK_COLS);
    rc = test_transpose_flt_blocked();
    ret |= rc;
    printf("%s\n", rc ? "Failed" : "Success");

    printf("\ntranspose_dbl_naive:\n");
    rc = test_transpose_dbl_naive();
    ret |= rc;
    printf("%s\n", rc ? "Failed" : "Success");

    printf("\ntranspose_dbl_blocked (block size = %zux%zu):\n",
           (size_t) TEST_BLK_ROWS, (size_t) TEST_BLK_COLS);
    rc = test_transpose_dbl_blocked();
    ret |= rc;
    printf("%s\n", rc ? "Failed" : "Success");

    printf("\ntranspose_fftw_complex:\n");
    rc = test_transpose_fftw_complex_naive();
    ret |= rc;
    printf("%s\n", rc ? "Failed" : "Success");

    printf("\ntranspose_flt_mkl:\n");
    rc = test_transpose_flt_mkl();
    ret |= rc;
    printf("%s\n", rc ? "Failed" : "Success");

    printf("\ntranspose_dbl_mkl:\n");
    rc = test_transpose_dbl_mkl();
    ret |= rc;
    printf("%s\n", rc ? "Failed" : "Success");

    printf("\ntranspose_cmplx8_mkl:\n");
    rc = test_transpose_cmplx8_mkl();
    ret |= rc;
    printf("%s\n", rc ? "Failed" : "Success");

    printf("\ntranspose_cmplx16_mkl:\n");
    rc = test_transpose_cmplx16_mkl();
    ret |= rc;
    printf("%s\n", rc ? "Failed" : "Success");
    return ret;
}
