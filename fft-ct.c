/**
 * FFT Corner Turn benchmark.
 *
 * 1-D FFTs -> Transpose -> 1-D FFTs
 *
 * @author Connor Imes <cimes@isi.edu>
 * @date 2019-07-15 
 */
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>

#include <fftw3.h>

#include "transpose.h"
#include "util.h"

static void *assert_fftw_malloc(size_t sz)
{
    void *ptr = fftw_malloc(sz);
    if (!ptr) {
        perror("fftw_malloc");
        exit(ENOMEM);
    }
    return ptr;
}

static void data_alloc(fftw_complex **A, fftw_complex **B, fftw_plan **p,
                       size_t nrows, size_t ncols)
{
    size_t i;
    *A = assert_fftw_malloc(nrows * ncols * sizeof(fftw_complex));
    *B = assert_fftw_malloc(nrows * ncols * sizeof(fftw_complex));
    *p = assert_fftw_malloc(nrows * sizeof(fftw_plan));
    for (i = 0; i < nrows; i++)
        (*p)[i] = fftw_plan_dft_1d(ncols,
                                   &(*A)[i * ncols], &(*B)[i * ncols],
                                   FFTW_FORWARD, FFTW_ESTIMATE);
}

static void data_free(fftw_complex *A, fftw_complex *B, fftw_plan *p,
                      size_t nrows)
{
    size_t i;
    for (i = 0; i < nrows; i++)
        fftw_destroy_plan(p[i]);
    fftw_free(p);
    fftw_free(B);
    fftw_free(A);
}

static void fft_tr_fft_1d(const fftw_plan *p1, const fftw_plan *p2,
                          fftw_complex *fft1_out, fftw_complex *fft2_in,
                          size_t fft1_rows, size_t fft1_cols)
{
    size_t i;
    // Perform first set of 1D FFTs
    for (i = 0; i < fft1_rows; i++)
        fftw_execute(p1[i]);
    // Matrix transpose
    transpose(fft1_out, fft2_in, fft1_rows, fft1_cols);
    // Perform second set of 1D FFTs
    for (i = 0; i < fft1_cols; i++)
        fftw_execute(p2[i]);
}

static void fft_ct_1d(size_t nrows, size_t ncols)
{
    fftw_complex *mat_fft1_in, *mat_fft1_out, *mat_fft2_in, *mat_fft2_out;
    fftw_plan *p_fft1, *p_fft2;

    // Setup FFT 1 (before transpose) and FFT 2 (after transpose)
    data_alloc(&mat_fft1_in, &mat_fft1_out, &p_fft1, nrows, ncols);
    data_alloc(&mat_fft2_in, &mat_fft2_out, &p_fft2, ncols, nrows);

    // Populate input with random data
    complex_fill_rand(mat_fft1_in, nrows * ncols);

    // Execute FFT 1 -> Transpose -> FFT2
    fft_tr_fft_1d(p_fft1, p_fft2, mat_fft1_out, mat_fft2_in, nrows, ncols);

    // Cleanup
    data_free(mat_fft2_in, mat_fft2_out, p_fft2, nrows);
    data_free(mat_fft1_in, mat_fft1_out, p_fft1, ncols);
}

static void usage(void)
{
    fprintf(stderr, "Usage: fft-ct <nrows> <ncols>\n");
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
    fft_ct_1d(nrows, ncols);
    return 0;
}
