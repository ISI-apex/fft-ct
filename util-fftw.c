/**
 * Utility functions
 *
 * @author Connor Imes <cimes@isi.edu>
 * @date 2019-07-15 
 */
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>

#include <fftw3.h>

#include "util.h"
#include "util-fftw.h"

#define FILL_RAND_COMPLEX(a, len, fn_rand) \
{ \
    size_t i; \
    for (i = 0; i < len; i++) { \
        a[i][0] = fn_rand(); /* re */ \
        a[i][1] = fn_rand(); /* im */ \
    } \
}

void fill_rand_fftwf_complex(fftwf_complex *a, size_t len)
{
    FILL_RAND_COMPLEX(a, len, rand_flt);
}

void fill_rand_fftw_complex(fftw_complex *a, size_t len)
{
    FILL_RAND_COMPLEX(a, len, rand_dbl);
}

#define MATRIX_PRINT_FFTW(A, nrows, ncols) \
{ \
    size_t r, c, i; \
    for (r = 0; r < nrows; r++) { \
        for (c = 0; c < ncols; c++) { \
            i = r * ncols + c; \
            printf("%s(%f, %f)", (c > 0 ? ", " : ""), A[i][0], A[i][1]); \
        } \
        printf("\n"); \
    } \
}

void matrix_print_fftwf_complex(fftwf_complex *A, size_t nrows, size_t ncols)
{
    MATRIX_PRINT_FFTW(A, nrows, ncols);
}

void matrix_print_fftw_complex(fftw_complex *A, size_t nrows, size_t ncols)
{
    MATRIX_PRINT_FFTW(A, nrows, ncols);
}

#define IS_EQ_CMPLX(a, b, fn_is_eq) \
    (fn_is_eq(a[0], b[0]) && fn_is_eq(a[1], b[1]))

int is_eq_fftwf_complex(const fftwf_complex a, const fftwf_complex b)
{
    return IS_EQ_CMPLX(a, b, is_eq_flt);
}

int is_eq_fftw_complex(const fftw_complex a, const fftw_complex b)
{
    return IS_EQ_CMPLX(a, b, is_eq_dbl);
}

void *assert_fftw_malloc(size_t sz)
{
    void *ptr = fftw_malloc(sz);
    if (!ptr) {
        perror("fftw_malloc");
        exit(ENOMEM);
    }
    return ptr;
}
