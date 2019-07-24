/**
 * FFT Corner Turn benchmark.
 *
 * 2-D FFT
 *
 * @author Connor Imes <cimes@isi.edu>
 * @date 2019-07-24
 */
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>

#include <fftw3.h>

#include "util.h"

static void data_alloc(fftw_complex **A, fftw_complex **B, fftw_plan *p,
                       size_t nrows, size_t ncols)
{
    *A = assert_fftw_malloc(nrows * ncols * sizeof(fftw_complex));
    *B = assert_fftw_malloc(nrows * ncols * sizeof(fftw_complex));
    *p = fftw_plan_dft_2d(nrows, ncols, *A, *B, FFTW_FORWARD, FFTW_ESTIMATE);
}

static void data_free(fftw_complex *A, fftw_complex *B, fftw_plan p)
{
    fftw_destroy_plan(p);
    fftw_free(B);
    fftw_free(A);
}

static void fft_2d(size_t nrows, size_t ncols)
{
    fftw_complex *mat_in, *mat_out;
    fftw_plan p;
    data_alloc(&mat_in, &mat_out, &p, nrows, ncols);
    // Populate input with random data
    complex_fill_rand(mat_in, nrows * ncols);
    fftw_execute(p);
    data_free(mat_in, mat_out, p);
}

static void usage(void)
{
    fprintf(stderr, "Usage: fft-2d <nrows> <ncols>\n");
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
    fft_2d(nrows, ncols);
    return 0;
}
