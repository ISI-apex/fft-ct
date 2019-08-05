/**
 * FFT Corner Turn benchmark.
 *
 * Transpose
 *
 * @author Connor Imes <cimes@isi.edu>
 * @date 2019-07-24
 */
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>

#include "transpose.h"
#include "util.h"

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

#define TRANSP_SETUP(datatype, fn_malloc, fn_fill, nrows, ncols) \
    datatype *A = fn_malloc(nrows * ncols * sizeof(datatype)); \
    datatype *B = fn_malloc(nrows * ncols * sizeof(datatype)); \
    fn_fill(A, nrows * ncols);

#define TRANSP_TEARDOWN(A, B, fn_free) \
    fn_free(B); \
    fn_free(A);

#define TRANSP(datatype, fn_malloc, fn_free, fn_fill, fn_transp, nrows, ncols) { \
    TRANSP_SETUP(datatype, fn_malloc, fn_fill, nrows, ncols); \
    fn_transp(A, B, nrows, ncols); \
    TRANSP_TEARDOWN(A, B, fn_free); \
}

#define TRANSP_BLOCKED(datatype, fn_malloc, fn_free, fn_fill, fn_transp, nrows, ncols, nblkrows, nblkcols) { \
    TRANSP_SETUP(datatype, fn_malloc, fn_fill, nrows, ncols); \
    fn_transp(A, B, nrows, ncols, nblkrows, nblkcols); \
    TRANSP_TEARDOWN(A, B, fn_free); \
}

static void usage(void)
{
    fprintf(stderr, "Usage: transp <nrows> <ncols>\n");
    exit(EINVAL);
}

int main(int argc, char **argv)
{
    size_t nrows, ncols;
    if (argc < 3)
        usage();
    nrows = atoi(argv[1]);
    ncols = atoi(argv[2]);
    if (!nrows || !ncols) {
        fprintf(stderr, "Parameters nrows and ncols must be > 0\n");
        return EINVAL;
    }

#if defined(USE_FLOAT_BLOCKED) || defined (USE_DOUBLE_BLOCKED)
    size_t nblkrows, nblkcols;
    if (argc > 4) {
        nblkrows = atoi(argv[3]);
        nblkrows = atoi(argv[4]);
    } else {
        nblkrows = nrows;
        nblkcols = ncols;
    }
#endif

#if defined(USE_FLOAT_NAIVE)
    TRANSP(float, assert_malloc, free,
           fill_rand_flt, transpose_flt_naive,
           nrows, ncols);
#elif defined(USE_DOUBLE_NAIVE)
    TRANSP(double, assert_malloc, free,
           fill_rand_dbl, transpose_dbl_naive,
           nrows, ncols);
#elif defined(USE_FLOAT_BLOCKED)
    TRANSP_BLOCKED(float, assert_malloc, free,
                   fill_rand_flt, transpose_flt_blocked,
                   nrows, ncols, nblkrows, nblkcols);
#elif defined(USE_DOUBLE_BLOCKED)
    TRANSP_BLOCKED(double, assert_malloc, free,
                   fill_rand_dbl, transpose_dbl_blocked,
                   nrows, ncols, nblkrows, nblkcols);
#elif defined(USE_FFTW_NAIVE)
    TRANSP(fftw_complex, assert_fftw_malloc, fftw_free,
           fill_rand_fftw_complex, transpose_fftw_complex_naive,
           nrows, ncols);
#elif defined(USE_MKL_FLOAT)
    TRANSP(float, assert_malloc, free,
           fill_rand_flt, transpose_flt_mkl,
           nrows, ncols);
#elif defined(USE_MKL_DOUBLE)
    TRANSP(double, assert_malloc, free,
           fill_rand_dbl, transpose_dbl_mkl,
           nrows, ncols);
#elif defined(USE_MKL_CMPLX8)
    TRANSP(MKL_Complex8, assert_malloc, free,
           fill_rand_cmplx8, transpose_cmplx8_mkl,
           nrows, ncols);
#elif defined(USE_MKL_CMPLX16)
    TRANSP(MKL_Complex16, assert_malloc, free,
           fill_rand_cmplx16, transpose_cmplx16_mkl,
           nrows, ncols);
#else
    #error "No matching transpose implementation found!"
#endif
    return 0;
}
