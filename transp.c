/**
 * FFT Corner Turn benchmark.
 *
 * Transpose
 *
 * @author Connor Imes <cimes@isi.edu>
 * @date 2019-07-24
 */
#include <complex.h>
#include <errno.h>
#include <getopt.h>
#include <limits.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "ptime.h"
#include "transpose.h"
#include "transpose-avx.h"
#include "transpose-threads.h"
#include "transpose-threads-avx.h"
#include "util.h"

#if defined(USE_FLT_BLOCKED) || \
    defined(USE_DBL_BLOCKED) || \
    defined(USE_FCMPLX_BLOCKED) || \
    defined(USE_DCMPLX_BLOCKED) || \
    defined(USE_FLT_THRROW_BLOCKED) || \
    defined(USE_DBL_THRROW_BLOCKED) || \
    defined(USE_FLT_THRCOL_BLOCKED) || \
    defined(USE_DBL_THRCOL_BLOCKED) || \
    defined(USE_FCMPLX_THRROW_BLOCKED) || \
    defined(USE_DCMPLX_THRROW_BLOCKED) || \
    defined(USE_FCMPLX_THRCOL_BLOCKED) || \
    defined(USE_DCMPLX_THRCOL_BLOCKED) || \
    defined(USE_FFTWF_BLOCKED) || \
    defined(USE_FFTW_BLOCKED) || \
    defined(USE_FFTWF_THRROW_BLOCKED) || \
    defined(USE_FFTWF_THRCOL_BLOCKED) || \
    defined(USE_FFTW_THRROW_BLOCKED) || \
    defined(USE_FFTW_THRCOL_BLOCKED)
#define _USE_TRANSP_BLOCKED 1
#endif

#if defined(USE_FLT_THRROW) || \
    defined(USE_DBL_THRROW) || \
    defined(USE_FLT_THRCOL) || \
    defined(USE_DBL_THRCOL) || \
    defined(USE_FCMPLX_THRROW) || \
    defined(USE_DCMPLX_THRROW) || \
    defined(USE_FCMPLX_THRCOL) || \
    defined(USE_DCMPLX_THRCOL) || \
    defined(USE_FLT_THRROW_BLOCKED) || \
    defined(USE_DBL_THRROW_BLOCKED) || \
    defined(USE_FLT_THRCOL_BLOCKED) || \
    defined(USE_DBL_THRCOL_BLOCKED) || \
    defined(USE_FCMPLX_THRROW_BLOCKED) || \
    defined(USE_DCMPLX_THRROW_BLOCKED) || \
    defined(USE_FCMPLX_THRCOL_BLOCKED) || \
    defined(USE_DCMPLX_THRCOL_BLOCKED) || \
    defined(USE_FFTWF_THRROW) || \
    defined(USE_FFTWF_THRCOL) || \
    defined(USE_FFTWF_THRROW_BLOCKED) || \
    defined(USE_FFTWF_THRCOL_BLOCKED) || \
    defined(USE_FFTW_THRROW) || \
    defined(USE_FFTW_THRCOL) || \
    defined(USE_FFTW_THRROW_BLOCKED) || \
    defined(USE_FFTW_THRCOL_BLOCKED) || \
    defined(USE_DBL_THRROW_AVX512_INTR) || \
    defined(USE_DBL_THRCOL_AVX512_INTR) || \
    defined(USE_FFTWF_THRROW_AVX512_INTR) || \
    defined(USE_FFTWF_THRCOL_AVX512_INTR)
#define _USE_TRANSP_THREADS 1
#endif

#if defined(USE_FFTWF_NAIVE) || \
    defined(USE_FFTWF_BLOCKED) || \
    defined(USE_FFTWF_THRROW) || \
    defined(USE_FFTWF_THRCOL) || \
    defined(USE_FFTWF_THRROW_BLOCKED) || \
    defined(USE_FFTWF_THRCOL_BLOCKED) || \
    defined(USE_FFTWF_AVX512_INTR) || \
    defined(USE_FFTWF_THRROW_AVX512_INTR) || \
    defined(USE_FFTWF_THRCOL_AVX512_INTR)
#include <fftw3.h>
#include "transpose-fftwf.h"
#include "transpose-fftwf-avx.h"
#include "transpose-fftwf-threads.h"
#include "transpose-fftwf-threads-avx.h"
#include "util-fftwf.h"
#endif
#if defined(USE_FFTW_NAIVE) || \
    defined(USE_FFTW_BLOCKED) || \
    defined(USE_FFTW_THRROW) || \
    defined(USE_FFTW_THRCOL) || \
    defined(USE_FFTW_THRROW_BLOCKED) || \
    defined(USE_FFTW_THRCOL_BLOCKED)
#include <fftw3.h>
#include "transpose-fftw.h"
#include "transpose-fftw-threads.h"
#include "util-fftw.h"
#endif

#if defined(USE_MKL_FLOAT) || defined(USE_MKL_DOUBLE) || \
    defined(USE_MKL_CMPLX8) || defined(USE_MKL_CMPLX16)
#include <mkl.h>
#include "transpose-mkl.h"
#include "util-mkl.h"
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

static bool do_print = false;
static bool do_verify = false;
static int rc = 0;

#define PRINT_ELAPSED_TIME(prefix, t1, t2) \
    printf("%s (ms): %f\n", prefix, ptime_elapsed_ns(t1, t2) / 1000000.0);

#define VERIFY_TRANSPOSE(A, B, fn_is_eq) { \
    size_t r, c; \
    for (r = 0; r < nrows && !rc; r++) { \
        for (c = 0; c < ncols && !rc; c++) { \
            rc = !fn_is_eq(A[r * ncols + c], B[c * nrows + r]); \
        } \
    } \
}

#define TRANSP_SETUP(datatype, fn_malloc, fn_fill, fn_mat_print) \
    datatype *A = fn_malloc(nrows * ncols * sizeof(datatype)); \
    datatype *B = fn_malloc(nrows * ncols * sizeof(datatype)); \
    ptime_gettime_monotonic(&t1); \
    fn_fill(A, nrows * ncols); \
    ptime_gettime_monotonic(&t2); \
    PRINT_ELAPSED_TIME("fill", &t1, &t2); \
    if (do_print) { \
        ptime_gettime_monotonic(&t1); \
        printf("In:\n"); \
        fn_mat_print(A, nrows, ncols); \
        ptime_gettime_monotonic(&t2); \
        PRINT_ELAPSED_TIME("print", &t1, &t2); \
    } \
    ptime_gettime_monotonic(&t1);

#define TRANSP_TEARDOWN(A, B, fn_mat_print, fn_is_eq, fn_free) \
    ptime_gettime_monotonic(&t2); \
    PRINT_ELAPSED_TIME("transpose", &t1, &t2); \
    if (do_print) { \
        printf("Out:\n"); \
        fn_mat_print(B, ncols, nrows); \
    } \
    if (do_verify) { \
        ptime_gettime_monotonic(&t1); \
        VERIFY_TRANSPOSE(A, B, fn_is_eq); \
        ptime_gettime_monotonic(&t2); \
        PRINT_ELAPSED_TIME("verify", &t1, &t2); \
    } \
    fn_free(B); \
    fn_free(A);

#define TRANSP(datatype, fn_malloc, fn_free, fn_fill, fn_mat_print, fn_transp, \
               fn_is_eq) { \
    TRANSP_SETUP(datatype, fn_malloc, fn_fill, fn_mat_print); \
    fn_transp(A, B, nrows, ncols); \
    TRANSP_TEARDOWN(A, B, fn_mat_print, fn_is_eq, fn_free); \
}

#define TRANSP_BLOCKED(datatype, fn_malloc, fn_free, fn_fill, fn_mat_print, \
                       fn_transp, fn_is_eq) { \
    TRANSP_SETUP(datatype, fn_malloc, fn_fill, fn_mat_print); \
    fn_transp(A, B, nrows, ncols, nblkrows, nblkcols); \
    TRANSP_TEARDOWN(A, B, fn_mat_print, fn_is_eq, fn_free); \
}

#define TRANSP_THREADED(datatype, fn_malloc, fn_free, fn_fill, fn_mat_print, \
                        fn_transp, fn_is_eq) { \
    TRANSP_SETUP(datatype, fn_malloc, fn_fill, fn_mat_print); \
    fn_transp(A, B, nrows, ncols, nthreads); \
    TRANSP_TEARDOWN(A, B, fn_mat_print, fn_is_eq, fn_free); \
}

#define TRANSP_THREADED_BLOCKED(datatype, fn_malloc, fn_free, fn_fill, \
                                fn_mat_print, fn_transp, fn_is_eq) { \
    TRANSP_SETUP(datatype, fn_malloc, fn_fill, fn_mat_print); \
    fn_transp(A, B, nrows, ncols, nthreads, nblkrows, nblkcols); \
    TRANSP_TEARDOWN(A, B, fn_mat_print, fn_is_eq, fn_free); \
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
            " [-p] [-v] [-h]\n"
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
            "  -p, --print              Print matrices\n"
            "  -v, --verify             Verify transpose\n"
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

static const char opts_short[] = "r:c:R:C:t:pvh";
static const struct option opts_long[] = {
    {"rows",        required_argument,  NULL,   'r'},
    {"cols",        required_argument,  NULL,   'c'},
    {"block-rows",  required_argument,  NULL,   'R'},
    {"block-cols",  required_argument,  NULL,   'C'},
    {"threads",     required_argument,  NULL,   't'},
    {"print",       no_argument,        NULL,   'p'},
    {"verify",      no_argument,        NULL,   'v'},
    {"help",        no_argument,        NULL,   'h'},
    {0, 0, 0, 0}
};

static void parse_args(int argc, char **argv)
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
        case 'p':
            do_print = true;
            break;
        case 'v':
            do_verify = true;
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
}

int main(int argc, char **argv)
{
    parse_args(argc, argv);
#if defined(USE_FLT_NAIVE)
    TRANSP(float, assert_malloc_al, free,
           fill_rand_flt, matrix_print_flt, transpose_flt_naive, is_eq_flt);
#elif defined(USE_DBL_NAIVE)
    TRANSP(double, assert_malloc_al, free,
           fill_rand_dbl, matrix_print_dbl, transpose_dbl_naive, is_eq_dbl);
#elif defined(USE_FCMPLX_NAIVE)
    TRANSP(float complex, assert_malloc_al, free,
           fill_rand_fcmplx, matrix_print_fcmplx, transpose_fcmplx_naive,
           is_eq_fcmplx);
#elif defined(USE_DCMPLX_NAIVE)
    TRANSP(double complex, assert_malloc_al, free,
           fill_rand_dcmplx, matrix_print_dcmplx, transpose_dcmplx_naive,
           is_eq_dcmplx);
#elif defined(USE_FLT_BLOCKED)
    TRANSP_BLOCKED(float, assert_malloc_al, free,
                   fill_rand_flt, matrix_print_flt, transpose_flt_blocked,
                   is_eq_flt);
#elif defined(USE_DBL_BLOCKED)
    TRANSP_BLOCKED(double, assert_malloc_al, free,
                   fill_rand_dbl, matrix_print_dbl, transpose_dbl_blocked,
                   is_eq_dbl);
#elif defined(USE_FCMPLX_BLOCKED)
    TRANSP_BLOCKED(float complex, assert_malloc_al, free,
                   fill_rand_fcmplx, matrix_print_fcmplx,
                   transpose_fcmplx_blocked, is_eq_fcmplx);
#elif defined(USE_DCMPLX_BLOCKED)
    TRANSP_BLOCKED(double complex, assert_malloc_al, free,
                   fill_rand_dcmplx, matrix_print_dcmplx,
                   transpose_dcmplx_blocked, is_eq_dcmplx);
#elif defined(USE_FLT_THRROW)
    TRANSP_THREADED(float, assert_malloc_al, free,
                    fill_rand_flt, matrix_print_flt, transpose_flt_threads_row,
                    is_eq_flt);
#elif defined(USE_DBL_THRROW)
    TRANSP_THREADED(double, assert_malloc_al, free,
                    fill_rand_dbl, matrix_print_dbl, transpose_dbl_threads_row,
                    is_eq_dbl);
#elif defined(USE_FLT_THRCOL)
    TRANSP_THREADED(float, assert_malloc_al, free,
                    fill_rand_flt, matrix_print_flt, transpose_flt_threads_col,
                    is_eq_flt);
#elif defined(USE_DBL_THRCOL)
    TRANSP_THREADED(double, assert_malloc_al, free,
                    fill_rand_dbl, matrix_print_dbl, transpose_dbl_threads_col,
                    is_eq_dbl);
#elif defined(USE_FCMPLX_THRROW)
    TRANSP_THREADED(float complex, assert_malloc_al, free,
                    fill_rand_fcmplx, matrix_print_fcmplx,
                    transpose_fcmplx_threads_row, is_eq_fcmplx);
#elif defined(USE_DCMPLX_THRROW)
    TRANSP_THREADED(double complex, assert_malloc_al, free,
                    fill_rand_dcmplx, matrix_print_dcmplx,
                    transpose_dcmplx_threads_row, is_eq_dcmplx);
#elif defined(USE_FCMPLX_THRCOL)
    TRANSP_THREADED(float complex, assert_malloc_al, free,
                    fill_rand_fcmplx, matrix_print_fcmplx,
                    transpose_fcmplx_threads_col, is_eq_fcmplx);
#elif defined(USE_DCMPLX_THRCOL)
    TRANSP_THREADED(double complex, assert_malloc_al, free,
                    fill_rand_dcmplx, matrix_print_dcmplx,
                    transpose_dcmplx_threads_col, is_eq_dcmplx);
#elif defined(USE_FLT_THRROW_BLOCKED)
    TRANSP_THREADED_BLOCKED(float, assert_malloc_al, free,
                            fill_rand_flt, matrix_print_flt,
                            transpose_flt_threads_row_blocked, is_eq_flt);
#elif defined(USE_DBL_THRROW_BLOCKED)
    TRANSP_THREADED_BLOCKED(double, assert_malloc_al, free,
                            fill_rand_dbl, matrix_print_dbl,
                            transpose_dbl_threads_row_blocked, is_eq_dbl);
#elif defined(USE_FLT_THRCOL_BLOCKED)
    TRANSP_THREADED_BLOCKED(float, assert_malloc_al, free,
                            fill_rand_flt, matrix_print_flt,
                            transpose_flt_threads_col_blocked, is_eq_flt);
#elif defined(USE_DBL_THRCOL_BLOCKED)
    TRANSP_THREADED_BLOCKED(double, assert_malloc_al, free,
                            fill_rand_dbl, matrix_print_dbl,
                            transpose_dbl_threads_col_blocked, is_eq_dbl);
#elif defined(USE_FCMPLX_THRROW_BLOCKED)
    TRANSP_THREADED_BLOCKED(float complex, assert_malloc_al, free,
                            fill_rand_fcmplx, matrix_print_fcmplx,
                            transpose_fcmplx_threads_row_blocked, is_eq_fcmplx);
#elif defined(USE_DCMPLX_THRROW_BLOCKED)
    TRANSP_THREADED_BLOCKED(double complex, assert_malloc_al, free,
                            fill_rand_dcmplx, matrix_print_dcmplx,
                            transpose_dcmplx_threads_row_blocked, is_eq_dcmplx);
#elif defined(USE_FCMPLX_THRCOL_BLOCKED)
    TRANSP_THREADED_BLOCKED(float complex, assert_malloc_al, free,
                            fill_rand_fcmplx, matrix_print_fcmplx,
                            transpose_fcmplx_threads_col_blocked, is_eq_fcmplx);
#elif defined(USE_DCMPLX_THRCOL_BLOCKED)
    TRANSP_THREADED_BLOCKED(double complex, assert_malloc_al, free,
                            fill_rand_dcmplx, matrix_print_dcmplx,
                            transpose_dcmplx_threads_col_blocked, is_eq_dcmplx);
#elif defined(USE_FFTWF_NAIVE)
    TRANSP(fftwf_complex, assert_fftwf_malloc, fftwf_free,
           fill_rand_fftwf, matrix_print_fftwf, transpose_fftwf_naive,
           is_eq_fftwf);
#elif defined(USE_FFTWF_BLOCKED)
    TRANSP_BLOCKED(fftwf_complex, assert_fftwf_malloc, fftwf_free,
                   fill_rand_fftwf, matrix_print_fftwf,
                   transpose_fftwf_blocked, is_eq_fftwf);
#elif defined(USE_FFTW_NAIVE)
    TRANSP(fftw_complex, assert_fftw_malloc, fftw_free,
           fill_rand_fftw, matrix_print_fftw, transpose_fftw_naive, is_eq_fftw);
#elif defined(USE_FFTW_BLOCKED)
    TRANSP_BLOCKED(fftw_complex, assert_fftw_malloc, fftw_free,
                   fill_rand_fftw, matrix_print_fftw,
                   transpose_fftw_blocked, is_eq_fftw);
#elif defined(USE_FFTWF_THRROW)
    TRANSP_THREADED(fftwf_complex, assert_fftwf_malloc, fftwf_free,
                    fill_rand_fftwf, matrix_print_fftwf,
                    transpose_fftwf_threads_row, is_eq_fftwf);
#elif defined(USE_FFTWF_THRCOL)
    TRANSP_THREADED(fftwf_complex, assert_fftwf_malloc, fftwf_free,
                    fill_rand_fftwf, matrix_print_fftwf,
                    transpose_fftwf_threads_col, is_eq_fftwf);
#elif defined(USE_FFTWF_THRROW_BLOCKED)
    TRANSP_THREADED_BLOCKED(fftwf_complex, assert_fftwf_malloc, fftwf_free,
                            fill_rand_fftwf, matrix_print_fftwf,
                            transpose_fftwf_threads_row_blocked, is_eq_fftwf);
#elif defined(USE_FFTWF_THRCOL_BLOCKED)
    TRANSP_THREADED_BLOCKED(fftwf_complex, assert_fftwf_malloc, fftwf_free,
                            fill_rand_fftwf, matrix_print_fftwf,
                            transpose_fftwf_threads_col_blocked, is_eq_fftwf);
#elif defined(USE_FFTW_THRROW)
    TRANSP_THREADED(fftw_complex, assert_fftw_malloc, fftw_free,
                    fill_rand_fftw, matrix_print_fftw,
                    transpose_fftw_threads_row, is_eq_fftw);
#elif defined(USE_FFTW_THRCOL)
    TRANSP_THREADED(fftw_complex, assert_fftw_malloc, fftw_free,
                    fill_rand_fftw, matrix_print_fftw,
                    transpose_fftw_threads_col, is_eq_fftw);
#elif defined(USE_FFTW_THRROW_BLOCKED)
    TRANSP_THREADED_BLOCKED(fftw_complex, assert_fftw_malloc, fftw_free,
                            fill_rand_fftw, matrix_print_fftw,
                            transpose_fftw_threads_row_blocked, is_eq_fftw);
#elif defined(USE_FFTW_THRCOL_BLOCKED)
    TRANSP_THREADED_BLOCKED(fftw_complex, assert_fftw_malloc, fftw_free,
                            fill_rand_fftw, matrix_print_fftw,
                            transpose_fftw_threads_col_blocked, is_eq_fftw);
#elif defined(USE_MKL_FLOAT)
    TRANSP(float, assert_malloc_al, free,
           fill_rand_flt, matrix_print_flt, transpose_flt_mkl, is_eq_flt);
#elif defined(USE_MKL_DOUBLE)
    TRANSP(double, assert_malloc_al, free,
           fill_rand_dbl, matrix_print_dbl, transpose_dbl_mkl, is_eq_dbl);
#elif defined(USE_MKL_CMPLX8)
    TRANSP(MKL_Complex8, assert_malloc_al, free,
           fill_rand_cmplx8, matrix_print_cmplx8, transpose_cmplx8_mkl,
           is_eq_cmplx8);
#elif defined(USE_MKL_CMPLX16)
    TRANSP(MKL_Complex16, assert_malloc_al, free,
           fill_rand_cmplx16, matrix_print_cmplx16, transpose_cmplx16_mkl,
           is_eq_cmplx16);
#elif defined(USE_DBL_AVX512_INTR)
    TRANSP(double, assert_malloc_al, free,
           fill_rand_dbl, matrix_print_dbl, transpose_dbl_avx512_intr,
           is_eq_dbl);
#elif defined(USE_FFTWF_AVX512_INTR)
    TRANSP(fftwf_complex, assert_fftwf_malloc, fftwf_free,
           fill_rand_fftwf, matrix_print_fftwf,
           transpose_fftwf_avx512_intr, is_eq_fftwf);
#elif defined(USE_DBL_THRROW_AVX512_INTR)
    TRANSP_THREADED(double, assert_malloc_al, free,
                    fill_rand_dbl, matrix_print_dbl,
                    transpose_dbl_thrrow_avx512_intr, is_eq_dbl);
#elif defined(USE_DBL_THRCOL_AVX512_INTR)
    TRANSP_THREADED(double, assert_malloc_al, free,
                    fill_rand_dbl, matrix_print_dbl,
                    transpose_dbl_thrcol_avx512_intr, is_eq_dbl);
#elif defined(USE_FFTWF_THRROW_AVX512_INTR)
    TRANSP_THREADED(fftwf_complex, assert_fftwf_malloc, fftwf_free,
                    fill_rand_fftwf, matrix_print_fftwf,
                    transpose_fftwf_thrrow_avx512_intr, is_eq_fftwf);
#elif defined(USE_FFTWF_THRCOL_AVX512_INTR)
    TRANSP_THREADED(fftwf_complex, assert_fftwf_malloc, fftwf_free,
                    fill_rand_fftwf, matrix_print_fftwf,
                    transpose_fftwf_thrcol_avx512_intr, is_eq_fftwf);
#else
    #error "No matching transpose implementation found!"
#endif
    return rc;
}
