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

#include "transpose.h"
#include "util.h"

static void transp_fftw_complex_naive(size_t nrows, size_t ncols)
{
    fftw_complex *A = assert_fftw_malloc(nrows * ncols * sizeof(fftw_complex));
    fftw_complex *B = assert_fftw_malloc(nrows * ncols * sizeof(fftw_complex));
    fill_rand_fftw_complex(A, nrows * ncols);
    transpose_fftw_complex_naive(A, B, nrows, ncols);
    fftw_free(B);
    fftw_free(A);
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
    transp_fftw_complex_naive(nrows, ncols);
    return 0;
}
