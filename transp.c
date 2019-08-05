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

#include <fftw3.h>

#include "transpose-fftw.h"
#include "util.h"
#include "util-fftw.h"

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
#if defined(USE_FFTW_NAIVE)
    TRANSP(fftw_complex, assert_fftw_malloc, fftw_free,
           fill_rand_fftw_complex, transpose_fftw_complex_naive,
           nrows, ncols);
#endif
    return 0;
}
