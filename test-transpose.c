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

#if defined(USE_FLOAT_THREADS_ROW) || defined(USE_DOUBLE_THREADS_ROW) || \
    defined(USE_FLOAT_THREADS_COL) || defined(USE_DOUBLE_THREADS_COL)
#include "transpose-threads.h"
#endif

#if defined(USE_FFTW_NAIVE)
#include <fftw3.h>
#include "transpose-fftw.h"
#include "util-fftw.h"
#endif

#if defined(USE_MKL_FLOAT) || defined(USE_MKL_DOUBLE) || \
    defined(USE_MKL_CMPLX8) || defined(USE_MKL_CMPLX16)
#include <mkl.h>
#include "transpose-mkl.h"
#include "util-mkl.h"
#endif

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

#define CHECK_TRANSPOSE(A, B, fn_is_eq, rc) { \
    size_t r, c; \
    rc = 0; \
    for (r = 0; r < TEST_ROWS && !rc; r++) { \
        for (c = 0; c < TEST_COLS && !rc; c++) { \
            rc = !fn_is_eq(A[r][c], B[c][r]); \
        } \
    } \
}

#define TEST_TRANSPOSE_BEG(datatype, fn_fill, fn_mat_print) \
    datatype A[TEST_ROWS][TEST_COLS]; \
    datatype B[TEST_COLS][TEST_ROWS]; \
    fn_fill(&A[0][0], TEST_ROWS * TEST_COLS); \
    printf("In:\n"); \
    fn_mat_print(&A[0][0], TEST_ROWS, TEST_COLS);

#define TEST_TRANSPOSE_END(A, B, fn_mat_print, fn_is_eq, rc) \
    printf("Out:\n"); \
    fn_mat_print(&B[0][0], TEST_COLS, TEST_ROWS); \
    CHECK_TRANSPOSE(A, B, fn_is_eq, rc);

#define TEST_TRANSPOSE(datatype, fn_fill, fn_mat_print, fn_transpose, fn_is_eq, rc) { \
    TEST_TRANSPOSE_BEG(datatype, fn_fill, fn_mat_print); \
    fn_transpose(&A[0][0], &B[0][0], TEST_ROWS, TEST_COLS); \
    TEST_TRANSPOSE_END(A, B, fn_mat_print, fn_is_eq, rc); \
}

#define TEST_TRANSPOSE_BLOCKED(datatype, fn_fill, fn_mat_print, fn_transpose, fn_is_eq, rc) { \
    TEST_TRANSPOSE_BEG(datatype, fn_fill, fn_mat_print); \
    fn_transpose(&A[0][0], &B[0][0], TEST_ROWS, TEST_COLS, TEST_BLK_ROWS, TEST_BLK_COLS); \
    TEST_TRANSPOSE_END(A, B, fn_mat_print, fn_is_eq, rc); \
}

int main(void)
{
    int rc;

#if defined(USE_FLOAT_NAIVE)
    printf("transpose_flt_naive:\n");
    TEST_TRANSPOSE(float, fill_rand_flt, matrix_print_flt, transpose_flt_naive,
                   is_eq_flt, rc);
#elif defined(USE_DOUBLE_NAIVE)
    printf("\ntranspose_dbl_naive:\n");
    TEST_TRANSPOSE(double, fill_rand_dbl, matrix_print_dbl, transpose_dbl_naive,
                   is_eq_dbl, rc);
#elif defined(USE_FLOAT_BLOCKED)
    printf("\ntranspose_flt_blocked (block size = %zux%zu):\n",
           (size_t) TEST_BLK_ROWS, (size_t) TEST_BLK_COLS);
    TEST_TRANSPOSE_BLOCKED(float, fill_rand_flt, matrix_print_flt,
                           transpose_flt_blocked, is_eq_flt, rc);
#elif defined(USE_DOUBLE_BLOCKED)
    printf("\ntranspose_dbl_blocked (block size = %zux%zu):\n",
           (size_t) TEST_BLK_ROWS, (size_t) TEST_BLK_COLS);
    TEST_TRANSPOSE_BLOCKED(double, fill_rand_dbl, matrix_print_dbl,
                           transpose_dbl_blocked, is_eq_dbl, rc);
#elif defined(USE_FLOAT_THREADS_ROW)
    printf("transpose_flt_threads_row:\n");
    // TODO
    rc = -1;
#elif defined(USE_DOUBLE_THREADS_ROW)
    printf("\ntranspose_dbl_threads_row:\n");
    // TODO
    rc = -1;
#elif defined(USE_FLOAT_THREADS_COL)
    printf("transpose_flt_threads_col:\n");
    // TODO
    rc = -1;
#elif defined(USE_DOUBLE_THREADS_COL)
    printf("\ntranspose_dbl_threads_col:\n");
    // TODO
    rc = -1;
#elif defined(USE_FFTW_NAIVE)
    printf("\ntranspose_fftw_complex_naive:\n");
    TEST_TRANSPOSE(fftw_complex, fill_rand_fftw_complex,
                   matrix_print_fftw_complex, transpose_fftw_complex_naive,
                   is_eq_fftw_complex, rc);
#elif defined(USE_MKL_FLOAT)
    printf("\ntranspose_flt_mkl:\n");
    TEST_TRANSPOSE(float, fill_rand_flt, matrix_print_flt, transpose_flt_mkl,
                   is_eq_flt, rc);
#elif defined(USE_MKL_DOUBLE)
    printf("\ntranspose_dbl_mkl:\n");
    TEST_TRANSPOSE(double, fill_rand_dbl, matrix_print_dbl, transpose_dbl_mkl,
                   is_eq_dbl, rc);
#elif defined(USE_MKL_CMPLX8)
    printf("\ntranspose_cmplx8_mkl:\n");
    TEST_TRANSPOSE(MKL_Complex8, fill_rand_cmplx8, matrix_print_cmplx8,
                   transpose_cmplx8_mkl, is_eq_cmplx8, rc);
#elif defined(USE_MKL_CMPLX16)
    printf("\ntranspose_cmplx16_mkl:\n");
    TEST_TRANSPOSE(MKL_Complex16, fill_rand_cmplx16, matrix_print_cmplx16,
                   transpose_cmplx16_mkl, is_eq_cmplx16, rc);
#else
    #error "No matching transpose implementation found!"
#endif

    printf("%s\n", rc ? "Failed" : "Success");
    return rc;
}
