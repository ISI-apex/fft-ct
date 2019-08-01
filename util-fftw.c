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

void fill_rand_fftw_complex(fftw_complex *a, size_t len)
{
    size_t i;
    for (i = 0; i < len; i++) {
        a[i][0] = rand_dbl(); // re
        a[i][1] = rand_dbl(); // im
    }
}

void matrix_print_fftw_complex(fftw_complex *A, size_t nrows, size_t ncols)
{
    size_t r, c, i;
    for (r = 0; r < nrows; r++) {
        for (c = 0; c < ncols; c++) {
            i = r * ncols + c;
            printf("%s(%f, %f)", (c > 0 ? ", " : ""), A[i][0], A[i][1]);
        }
        printf("\n");
    }
}

int is_eq_fftw_complex(const fftw_complex a, const fftw_complex b)
{
    return is_eq_dbl(a[0], b[0]) && is_eq_dbl(a[1], b[1]);
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
