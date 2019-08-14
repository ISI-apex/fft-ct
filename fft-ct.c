/**
 * FFT Corner Turn benchmark.
 *
 * 1-D FFTs -> Transpose -> 1-D FFTs
 *
 * @author Connor Imes <cimes@isi.edu>
 * @date 2019-07-15
 */
#include <complex.h>
#include <errno.h>
#include <getopt.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>

#include <fftw3.h>

#include "transpose-fftw.h"
#include "util.h"
#include "util-fftw.h"

#if defined(USE_FFTW_BLOCKED)
#define _USE_TRANSP_BLOCKED 1
#endif

static size_t nrows = 0;
static size_t ncols = 0;

#if defined(_USE_TRANSP_BLOCKED)
static size_t nblkrows = 0;
static size_t nblkcols = 0;
#endif

static void data_alloc(fftw_complex **A, fftw_complex **B, fftw_plan **p,
                       size_t r, size_t c)
{
    size_t i;
    *A = assert_fftw_malloc(r * c * sizeof(fftw_complex));
    *B = assert_fftw_malloc(r * c * sizeof(fftw_complex));
    *p = assert_fftw_malloc(r * sizeof(fftw_plan));
    for (i = 0; i < r; i++)
        (*p)[i] = fftw_plan_dft_1d(c, &(*A)[i * c], &(*B)[i * c],
                                   FFTW_FORWARD, FFTW_ESTIMATE);
}

static void data_free(fftw_complex *A, fftw_complex *B, fftw_plan *p,
                      size_t r)
{
    size_t i;
    for (i = 0; i < r; i++)
        fftw_destroy_plan(p[i]);
    fftw_free(p);
    fftw_free(B);
    fftw_free(A);
}

static void fft_tr_fft_1d(const fftw_plan *p1, const fftw_plan *p2,
                          fftw_complex *fft1_out, fftw_complex *fft2_in)
{
    size_t i;
    // Perform first set of 1D FFTs
    for (i = 0; i < nrows; i++)
        fftw_execute(p1[i]);
    // Matrix transpose
#if defined(USE_FFTW_NAIVE)
    transpose_fftw_complex_naive(fft1_out, fft2_in, nrows, ncols);
#elif defined(USE_FFTW_BLOCKED)
    transpose_fftw_complex_blocked(fft1_out, fft2_in, nrows, ncols, nblkrows,
                                   nblkcols);
#else
    #error "No matching transpose implementation found!"
#endif
    // Perform second set of 1D FFTs
    for (i = 0; i < ncols; i++)
        fftw_execute(p2[i]);
}

static void fft_ct_1d(void)
{
    fftw_complex *mat_fft1_in, *mat_fft1_out, *mat_fft2_in, *mat_fft2_out;
    fftw_plan *p_fft1, *p_fft2;

    // Setup FFT 1 (before transpose) and FFT 2 (after transpose)
    data_alloc(&mat_fft1_in, &mat_fft1_out, &p_fft1, nrows, ncols);
    data_alloc(&mat_fft2_in, &mat_fft2_out, &p_fft2, ncols, nrows);

    // Populate input with random data
    fill_rand_fftw_complex(mat_fft1_in, nrows * ncols);

    // Execute FFT 1 -> Transpose -> FFT2
    fft_tr_fft_1d(p_fft1, p_fft2, mat_fft1_out, mat_fft2_in);

    // Cleanup
    data_free(mat_fft2_in, mat_fft2_out, p_fft2, ncols);
    data_free(mat_fft1_in, mat_fft1_out, p_fft1, nrows);
}

static void usage(const char *pname, int code)
{
    fprintf(code ? stderr : stdout,
            "Usage: %s -r ROWS -c COLS [-h]\n"
            "  -r, --rows=ROWS          Matrix row count, in [1, ULONG_MAX]\n"
            "  -c, --cols=COLS          Matrix column count, in [1, ULONG_MAX]\n"
#if defined(_USE_TRANSP_BLOCKED)
            "  -R, --block-rows=ROWS    Rows per block, in [0, ULONG_MAX]\n"
            "  -C, --block-cols=COLS    Columns per block, in [0, ULONG_MAX]\n"
            "                           ROWS/COLS must be divisors of the corresponding\n"
            "                           matrix dimension\n"
            "                           (default=0, implies no blocking in that dimension)\n"
#endif
            "  -h, --help               Print this message and exit\n",
            pname);
    exit(code);
}

static size_t assert_to_size_t(const char* str, const char* pname)
{
    size_t s = strtoul(str, NULL, 0);
    if (s == ULONG_MAX && errno == ERANGE) {
        usage(pname, errno);
    }
    return s;
}

static const char opts_short[] = "r:c:R:C:h";
static const struct option opts_long[] = {
    {"rows",        required_argument,  NULL,   'r'},
    {"cols",        required_argument,  NULL,   'c'},
    {"block-rows",  required_argument,  NULL,   'R'},
    {"block-cols",  required_argument,  NULL,   'C'},
    {"help",        no_argument,        NULL,   'h'},
    {0, 0, 0, 0}
};

int main(int argc, char **argv)
{
    int c;

    while ((c = getopt_long(argc, argv, opts_short, opts_long, NULL)) != -1) {
        switch (c) {
        case 'r':
            nrows = assert_to_size_t(optarg, argv[0]);
            break;
        case 'c':
            ncols = assert_to_size_t(optarg, argv[0]);
            break;
#if defined(_USE_TRANSP_BLOCKED)
        case 'R':
            nblkrows = assert_to_size_t(optarg, argv[0]);
            break;
        case 'C':
            nblkcols = assert_to_size_t(optarg, argv[0]);
            break;
#endif
        case 'h':
            usage(argv[0], 0);
            break;
        default:
            usage(argv[0], EINVAL);
            break;
        }
    }
    if (!nrows || !ncols) {
        usage(argv[0], EINVAL);
    }
#if defined(_USE_TRANSP_BLOCKED)
    // fall back to default values
    if (!nblkrows) {
        nblkrows = nrows;
    }
    if (!nblkcols) {
        nblkcols = ncols;
    }
    // check divisibility
    if ((nrows % nblkrows) || (ncols % nblkcols)) {
        usage(argv[0], EINVAL);
    }
#endif
    fft_ct_1d();
    return 0;
}
