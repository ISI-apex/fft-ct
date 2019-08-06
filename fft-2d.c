/**
 * FFT Corner Turn benchmark.
 *
 * 2-D FFT
 *
 * @author Connor Imes <cimes@isi.edu>
 * @date 2019-07-24
 */
#include <errno.h>
#include <getopt.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>

#include <fftw3.h>

#include "util-fftw.h"

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
    fill_rand_fftw_complex(mat_in, nrows * ncols);
    fftw_execute(p);
    data_free(mat_in, mat_out, p);
}

static void usage(const char *pname, int code)
{
    fprintf(code ? stderr : stdout,
            "Usage: %s -r ROWS -c COLS [-h]\n"
            "  -r, --rows=ROWS          Matrix row count, in [1, ULONG_MAX]\n"
            "  -c, --cols=COLS          Matrix column count, in [1, ULONG_MAX]\n"
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

static const char opts_short[] = "r:c:h";
static const struct option opts_long[] = {
    {"rows",        required_argument,  NULL,   'r'},
    {"cols",        required_argument,  NULL,   'c'},
    {"help",        no_argument,        NULL,   'h'},
    {0, 0, 0, 0}
};

int main(int argc, char **argv)
{
    size_t nrows = 0;
    size_t ncols = 0;
    int c;

    while ((c = getopt_long(argc, argv, opts_short, opts_long, NULL)) != -1) {
        switch (c) {
        case 'r':
            nrows = assert_to_size_t(optarg, argv[0]);
            break;
        case 'c':
            ncols = assert_to_size_t(optarg, argv[0]);
            break;
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
    fft_2d(nrows, ncols);
    return 0;
}
