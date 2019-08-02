/**
 * Utility functions
 *
 * @author Connor Imes <cimes@isi.edu>
 * @date 2019-07-15 
 */
#include <stdio.h>
#include <stdlib.h>

#include <mkl.h>

#include "util.h"
#include "util-mkl.h"

void fill_rand_cmplx8(MKL_Complex8 *a, size_t len)
{
    size_t i;
    for (i = 0; i < len; i++) {
        a[i].real = rand_flt();
        a[i].imag = rand_flt();
    }
}

void fill_rand_cmplx16(MKL_Complex16 *a, size_t len)
{
    size_t i;
    for (i = 0; i < len; i++) {
        a[i].real = rand_dbl();
        a[i].imag = rand_dbl();
    }
}

void matrix_print_cmplx8(const MKL_Complex8 *A, size_t nrows, size_t ncols)
{
    size_t r, c, i;
    for (r = 0; r < nrows; r++) {
        for (c = 0; c < ncols; c++) {
            i = r * ncols + c;
            printf("%s(%f, %f)", (c > 0 ? ", " : ""), A[i].real, A[i].imag);
        }
        printf("\n");
    }
}

void matrix_print_cmplx16(const MKL_Complex16 *A, size_t nrows, size_t ncols)
{
    size_t r, c, i;
    for (r = 0; r < nrows; r++) {
        for (c = 0; c < ncols; c++) {
            i = r * ncols + c;
            printf("%s(%f, %f)", (c > 0 ? ", " : ""), A[i].real, A[i].imag);
        }
        printf("\n");
    }
}

int is_eq_cmplx8(const MKL_Complex8 a, const MKL_Complex8 b)
{
    return is_eq_flt(a.real, b.real) && is_eq_flt(a.imag, b.imag);
}

int is_eq_cmplx16(const MKL_Complex16 a, const MKL_Complex16 b)
{
    return is_eq_dbl(a.real, b.real) && is_eq_dbl(a.imag, b.imag);
}
