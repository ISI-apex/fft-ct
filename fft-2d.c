/**
 * FFT Corner Turn benchmark.
 *
 * 2-D FFT
 *
 * @author Connor Imes <cimes@isi.edu>
 * @date 2019-07-24
 */
#include <complex.h>
#include <errno.h>
#include <getopt.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>

#include <fftw3.h>

#include "ptime.h"

#if defined(USE_FFTWF)
#include "util-fftwf.h"
typedef fftwf_complex       FFTW_COMPLEX_T;
typedef fftwf_plan          FFTW_PLAN_T;
#define ASSERT_FFTW_MALLOC  assert_fftwf_malloc
#define FFTW_FREE           fftwf_free
#define FFTW_PLAN_2D        fftwf_plan_dft_2d
#define FFTW_PLAN_DESTROY   fftwf_destroy_plan
#define FFTW_EXECUTE        fftwf_execute
#define FILL_RAND           fill_rand_fftwf
#else
#include "util-fftw.h"
typedef fftw_complex        FFTW_COMPLEX_T;
typedef fftw_plan           FFTW_PLAN_T;
#define ASSERT_FFTW_MALLOC  assert_fftw_malloc
#define FFTW_FREE           fftw_free
#define FFTW_PLAN_2D        fftw_plan_dft_2d
#define FFTW_PLAN_DESTROY   fftw_destroy_plan
#define FFTW_EXECUTE        fftw_execute
#define FILL_RAND           fill_rand_fftw
#endif

#define PRINT_ELAPSED_TIME(prefix, t1, t2) \
    printf("%s (ms): %f\n", prefix, ptime_elapsed_ns(t1, t2) / 1000000.0);

static void data_alloc(FFTW_COMPLEX_T **A, FFTW_COMPLEX_T **B, FFTW_PLAN_T *p,
                       size_t nrows, size_t ncols)
{
    *A = ASSERT_FFTW_MALLOC(nrows * ncols * sizeof(**A));
    *B = ASSERT_FFTW_MALLOC(nrows * ncols * sizeof(**B));
    *p = FFTW_PLAN_2D(nrows, ncols, *A, *B, FFTW_FORWARD, FFTW_ESTIMATE);
}

static void data_free(FFTW_COMPLEX_T *A, FFTW_COMPLEX_T *B, FFTW_PLAN_T p)
{
    FFTW_PLAN_DESTROY(p);
    FFTW_FREE(B);
    FFTW_FREE(A);
}

static void fft_2d(size_t nrows, size_t ncols)
{
    struct timespec t1, t2;
    FFTW_COMPLEX_T *mat_in, *mat_out;
    FFTW_PLAN_T p;
    data_alloc(&mat_in, &mat_out, &p, nrows, ncols);

    // Populate input with random data
    ptime_gettime_monotonic(&t1);
    FILL_RAND(mat_in, nrows * ncols);
    ptime_gettime_monotonic(&t2);
    PRINT_ELAPSED_TIME("fill", &t1, &t2);

    ptime_gettime_monotonic(&t1);
    FFTW_EXECUTE(p);
    ptime_gettime_monotonic(&t2);
    PRINT_ELAPSED_TIME("fft-2d", &t1, &t2);

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
