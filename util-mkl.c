/**
 * Utility functions
 *
 * @author Connor Imes <cimes@isi.edu>
 * @date 2019-07-15 
 */
#include <complex.h>
#include <stdio.h>
#include <stdlib.h>

#include <mkl.h>

#include "util.h"
#include "util-mkl.h"

void fill_rand_cmplx8(MKL_Complex8 *a, size_t len)
{
    fill_rand_fcmplx(a, len);
}

void fill_rand_cmplx16(MKL_Complex16 *a, size_t len)
{
    fill_rand_dcmplx(a, len);
}

void matrix_print_cmplx8(const MKL_Complex8 *A, size_t nrows, size_t ncols)
{
    matrix_print_fcmplx(A, nrows, ncols);
}

void matrix_print_cmplx16(const MKL_Complex16 *A, size_t nrows, size_t ncols)
{
    matrix_print_dcmplx(A, nrows, ncols);
}

int is_eq_cmplx8(const MKL_Complex8 a, const MKL_Complex8 b)
{
    return is_eq_fcmplx(a, b);
}

int is_eq_cmplx16(const MKL_Complex16 a, const MKL_Complex16 b)
{
    return is_eq_dcmplx(a, b);
}
