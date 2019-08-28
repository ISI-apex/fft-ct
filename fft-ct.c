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

#include "ptime.h"

#if defined(USE_FFTWF_NAIVE) || \
    defined(USE_FFTWF_BLOCKED) || \
    defined(USE_FFTWF_THRROW) || \
    defined(USE_FFTWF_THRCOL) || \
    defined(USE_FFTWF_THRROW_BLOCKED) || \
    defined(USE_FFTWF_THRCOL_BLOCKED) || \
    defined(USE_FFTWF_AVX512_INTR) || \
    defined(USE_FFTWF_THRROW_AVX512_INTR) || \
    defined(USE_FFTWF_THRCOL_AVX512_INTR)
#include "transpose-fftwf.h"
#include "transpose-fftwf-avx.h"
#include "transpose-fftwf-threads.h"
#include "transpose-fftwf-threads-avx.h"
#include "util-fftwf.h"
typedef fftwf_complex       FFTW_COMPLEX_T;
typedef fftwf_plan          FFTW_PLAN_T;
#define ASSERT_FFTW_MALLOC  assert_fftwf_malloc
#define FFTW_FREE           fftwf_free
#define FFTW_PLAN_1D        fftwf_plan_dft_1d
#define FFTW_PLAN_DESTROY   fftwf_destroy_plan
#define FFTW_EXECUTE        fftwf_execute
#define FILL_RAND           fill_rand_fftwf
#else
#include "transpose-fftw.h"
#include "transpose-fftw-threads.h"
#include "util-fftw.h"
typedef fftw_complex        FFTW_COMPLEX_T;
typedef fftw_plan           FFTW_PLAN_T;
#define ASSERT_FFTW_MALLOC  assert_fftw_malloc
#define FFTW_FREE           fftw_free
#define FFTW_PLAN_1D        fftw_plan_dft_1d
#define FFTW_PLAN_DESTROY   fftw_destroy_plan
#define FFTW_EXECUTE        fftw_execute
#define FILL_RAND           fill_rand_fftw
#endif

#if defined(USE_FFTWF_BLOCKED) || \
    defined(USE_FFTWF_THRROW_BLOCKED) || \
    defined(USE_FFTWF_THRCOL_BLOCKED) || \
    defined(USE_FFTW_BLOCKED) || \
    defined(USE_FFTW_THRROW_BLOCKED) || \
    defined(USE_FFTW_THRCOL_BLOCKED)
#define _USE_TRANSP_BLOCKED 1
#endif

#if defined(USE_FFTWF_THRROW) || \
    defined(USE_FFTWF_THRCOL) || \
    defined(USE_FFTWF_THRROW_BLOCKED) || \
    defined(USE_FFTWF_THRCOL_BLOCKED) || \
    defined(USE_FFTWF_THRROW_AVX512_INTR) || \
    defined(USE_FFTWF_THRCOL_AVX512_INTR) || \
    defined(USE_FFTW_THRROW) || \
    defined(USE_FFTW_THRCOL) || \
    defined(USE_FFTW_THRROW_BLOCKED) || \
    defined(USE_FFTW_THRCOL_BLOCKED)
#define _USE_TRANSP_THREADS 1
#endif

static size_t nrows = 0;
static size_t ncols = 0;
static struct timespec t1;
static struct timespec t2;

#if defined(_USE_TRANSP_BLOCKED)
static size_t nblkrows = 0;
static size_t nblkcols = 0;
#endif

#if defined(_USE_TRANSP_THREADS)
static size_t nthreads = 1;
#endif

#define PRINT_ELAPSED_TIME(prefix, t1, t2) \
    printf("%s (ms): %f\n", prefix, ptime_elapsed_ns(t1, t2) / 1000000.0);

static void data_alloc(FFTW_COMPLEX_T **A, FFTW_COMPLEX_T **B, FFTW_PLAN_T **p,
                       size_t r, size_t c)
{
    size_t i;
    *A = ASSERT_FFTW_MALLOC(r * c * sizeof(**A));
    *B = ASSERT_FFTW_MALLOC(r * c * sizeof(**B));
    *p = ASSERT_FFTW_MALLOC(r * sizeof(**p));
    for (i = 0; i < r; i++) {
        (*p)[i] = FFTW_PLAN_1D(c, &(*A)[i * c], &(*B)[i * c],
                               FFTW_FORWARD, FFTW_ESTIMATE);
    }
}

static void data_free(FFTW_COMPLEX_T *A, FFTW_COMPLEX_T *B, FFTW_PLAN_T *p,
                      size_t r)
{
    size_t i;
    for (i = 0; i < r; i++) {
        FFTW_PLAN_DESTROY(p[i]);
    }
    FFTW_FREE(p);
    FFTW_FREE(B);
    FFTW_FREE(A);
}

static void fft_tr_fft_1d(const FFTW_PLAN_T *p1, const FFTW_PLAN_T *p2,
                          FFTW_COMPLEX_T *fft1_out, FFTW_COMPLEX_T *fft2_in)
{
    size_t i;

    // Perform first set of 1D FFTs
    ptime_gettime_monotonic(&t1);
    for (i = 0; i < nrows; i++) {
        FFTW_EXECUTE(p1[i]);
    }
    ptime_gettime_monotonic(&t2);
    PRINT_ELAPSED_TIME("fft-1d-1", &t1, &t2);

    // Matrix transpose
    ptime_gettime_monotonic(&t1);
#if defined(USE_FFTWF_NAIVE)
    transpose_fftwf_naive(fft1_out, fft2_in, nrows, ncols);
#elif defined(USE_FFTWF_BLOCKED)
    transpose_fftwf_blocked(fft1_out, fft2_in, nrows, ncols, nblkrows,
                            nblkcols);
#elif defined(USE_FFTWF_THRROW)
    transpose_fftwf_threads_row(fft1_out, fft2_in, nrows, ncols, nthreads);
#elif defined(USE_FFTWF_THRCOL)
    transpose_fftwf_threads_col(fft1_out, fft2_in, nrows, ncols, nthreads);
#elif defined(USE_FFTWF_THRROW_BLOCKED)
    transpose_fftwf_threads_row_blocked(fft1_out, fft2_in, nrows, ncols,
                                        nthreads, nblkrows, nblkcols);
#elif defined(USE_FFTWF_THRCOL_BLOCKED)
    transpose_fftwf_threads_col_blocked(fft1_out, fft2_in, nrows, ncols,
                                        nthreads, nblkrows, nblkcols);
#elif defined(USE_FFTWF_AVX512_INTR)
    transpose_fftwf_avx512_intr(fft1_out, fft2_in, nrows, ncols);
#elif defined(USE_FFTWF_THRROW_AVX512_INTR)
    transpose_fftwf_thrrow_avx512_intr(fft1_out, fft2_in, nrows, ncols,
                                       nthreads);
#elif defined(USE_FFTWF_THRCOL_AVX512_INTR)
    transpose_fftwf_thrcol_avx512_intr(fft1_out, fft2_in, nrows, ncols,
                                       nthreads);
#elif defined(USE_FFTW_NAIVE)
    transpose_fftw_naive(fft1_out, fft2_in, nrows, ncols);
#elif defined(USE_FFTW_BLOCKED)
    transpose_fftw_blocked(fft1_out, fft2_in, nrows, ncols, nblkrows,
                           nblkcols);
#elif defined(USE_FFTW_THRROW)
    transpose_fftw_threads_row(fft1_out, fft2_in, nrows, ncols, nthreads);
#elif defined(USE_FFTW_THRCOL)
    transpose_fftw_threads_col(fft1_out, fft2_in, nrows, ncols, nthreads);
#elif defined(USE_FFTW_THRROW_BLOCKED)
    transpose_fftw_threads_row_blocked(fft1_out, fft2_in, nrows, ncols,
                                       nthreads, nblkrows, nblkcols);
#elif defined(USE_FFTW_THRCOL_BLOCKED)
    transpose_fftw_threads_col_blocked(fft1_out, fft2_in, nrows, ncols,
                                       nthreads, nblkrows, nblkcols);
#else
    #error "No matching transpose implementation found!"
#endif
    ptime_gettime_monotonic(&t2);
    PRINT_ELAPSED_TIME("transpose", &t1, &t2);

    ptime_gettime_monotonic(&t1);
    // Perform second set of 1D FFTs
    for (i = 0; i < ncols; i++) {
        FFTW_EXECUTE(p2[i]);
    }
    ptime_gettime_monotonic(&t2);
    PRINT_ELAPSED_TIME("fft-1d-2", &t1, &t2);
}

static void fft_ct_1d(void)
{
    FFTW_COMPLEX_T *mat_fft1_in, *mat_fft1_out, *mat_fft2_in, *mat_fft2_out;
    FFTW_PLAN_T *p_fft1, *p_fft2;

    // Setup FFT 1 (before transpose) and FFT 2 (after transpose)
    data_alloc(&mat_fft1_in, &mat_fft1_out, &p_fft1, nrows, ncols);
    data_alloc(&mat_fft2_in, &mat_fft2_out, &p_fft2, ncols, nrows);

    // Populate input with random data
    ptime_gettime_monotonic(&t1);
    FILL_RAND(mat_fft1_in, nrows * ncols);
    ptime_gettime_monotonic(&t2);
    PRINT_ELAPSED_TIME("fill", &t1, &t2);

    // Execute FFT 1 -> Transpose -> FFT2
    fft_tr_fft_1d(p_fft1, p_fft2, mat_fft1_out, mat_fft2_in);

    // Cleanup
    data_free(mat_fft2_in, mat_fft2_out, p_fft2, ncols);
    data_free(mat_fft1_in, mat_fft1_out, p_fft1, nrows);
}

static void usage(const char *pname, int code)
{
    fprintf(code ? stderr : stdout,
            "Usage: %s -r ROWS -c COLS"
#if defined(_USE_TRANSP_BLOCKED)
            " [-R ROWS] [-C COLS]"
#endif
#if defined(_USE_TRANSP_THREADS)
            " [-t THREADS]"
#endif
            " [-h]\n"
            "  -r, --rows=ROWS          Matrix row count, in [1, ULONG_MAX]\n"
            "  -c, --cols=COLS          Matrix column count, in [1, ULONG_MAX]\n"
#if defined(_USE_TRANSP_BLOCKED)
            "  -R, --block-rows=ROWS    Rows per block, in [0, ULONG_MAX]\n"
            "  -C, --block-cols=COLS    Columns per block, in [0, ULONG_MAX]\n"
            "                           ROWS/COLS must be divisors of the corresponding\n"
            "                           matrix dimension\n"
            "                           (default=0, implies no blocking in that dimension)\n"
#endif
#if defined(_USE_TRANSP_THREADS)
            "  -t, --threads=THREADS    Number of threads, in (0, ULONG_MAX] (default=1)\n"
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

static const char opts_short[] = "r:c:R:C:t:h";
static const struct option opts_long[] = {
    {"rows",        required_argument,  NULL,   'r'},
    {"cols",        required_argument,  NULL,   'c'},
    {"block-rows",  required_argument,  NULL,   'R'},
    {"block-cols",  required_argument,  NULL,   'C'},
    {"threads",     required_argument,  NULL,   't'},
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
#if defined(_USE_TRANSP_THREADS)
        case 't':
            nthreads = assert_to_size_t(optarg, argv[0]);
            if (!nthreads) {
                usage(argv[0], EINVAL);
            }
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
