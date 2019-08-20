/**
 * Transpose functions.
 *
 * @author Kaushik Datta <kdatta@isi.edu>
 * @author Connor Imes <cimes@isi.edu>
 * @date 2019-08-06
 */
#include <assert.h>
#include <complex.h>
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>

#include "transpose-threads.h"
#include "util.h"

struct tr_thread_arg {
    const void* restrict A;
    void* restrict B;
    size_t A_rows, A_cols;
    size_t r_min, r_max, c_min, c_max;
    size_t blk_rows, blk_cols;
    size_t thr_num;
};

static void tt_arg_init(struct tr_thread_arg *tt_arg,
                        const void* restrict A, void* restrict B,
                        size_t A_rows, size_t A_cols,
                        size_t r_min, size_t r_max, size_t c_min, size_t c_max,
                        size_t blk_rows, size_t blk_cols,
                        size_t thr_num)
{
    tt_arg->A = A;
    tt_arg->B = B;
    tt_arg->A_rows = A_rows;
    tt_arg->A_cols = A_cols;
    tt_arg->r_min = r_min;
    tt_arg->r_max = r_max;
    tt_arg->c_min = c_min;
    tt_arg->c_max = c_max;
    tt_arg->blk_rows = blk_rows;
    tt_arg->blk_cols = blk_cols;
    tt_arg->thr_num = thr_num;
}

#define TRANSPOSE_BLK(A, B, A_rows, A_cols, r_min, c_min, r_max, c_max) { \
    size_t r, c; \
    for (r = (r_min); r < (r_max); r++) { \
        for (c = (c_min); c < (c_max); c++) { \
            (B)[(c) * (A_rows) + (r)] = (A)[(r) * (A_cols) + (c)]; \
        } \
    } \
}

static void *transpose_thread_flt(void *args)
{
    const struct tr_thread_arg *tt_arg = (const struct tr_thread_arg *)args;
    TRANSPOSE_BLK((const float* restrict)tt_arg->A,
                  (float* restrict)tt_arg->B,
                  tt_arg->A_rows, tt_arg->A_cols,
                  tt_arg->r_min, tt_arg->c_min, tt_arg->r_max, tt_arg->c_max);
    pthread_exit((void *)tt_arg->thr_num);
}

static void *transpose_thread_dbl(void *args)
{
    const struct tr_thread_arg *tt_arg = (const struct tr_thread_arg *)args;
    TRANSPOSE_BLK((const double* restrict)tt_arg->A,
                  (double* restrict)tt_arg->B,
                  tt_arg->A_rows, tt_arg->A_cols,
                  tt_arg->r_min, tt_arg->c_min, tt_arg->r_max, tt_arg->c_max);
    pthread_exit((void *)tt_arg->thr_num);
}

static void *transpose_thread_flt_cmplx(void *args)
{
    const struct tr_thread_arg *tt_arg = (const struct tr_thread_arg *)args;
    TRANSPOSE_BLK((const float complex* restrict)tt_arg->A,
                  (float complex* restrict)tt_arg->B,
                  tt_arg->A_rows, tt_arg->A_cols,
                  tt_arg->r_min, tt_arg->c_min, tt_arg->r_max, tt_arg->c_max);
    pthread_exit((void *)tt_arg->thr_num);
}

static void *transpose_thread_dbl_cmplx(void *args)
{
    const struct tr_thread_arg *tt_arg = (const struct tr_thread_arg *)args;
    TRANSPOSE_BLK((const double complex* restrict)tt_arg->A,
                  (double complex* restrict)tt_arg->B,
                  tt_arg->A_rows, tt_arg->A_cols,
                  tt_arg->r_min, tt_arg->c_min, tt_arg->r_max, tt_arg->c_max);
    pthread_exit((void *)tt_arg->thr_num);
}

static void *transpose_thread_blocked_flt(void *args) {
    const struct tr_thread_arg *tt_arg = (const struct tr_thread_arg *)args;
    const size_t start_rblk_num = tt_arg->r_min / tt_arg->blk_rows;
    const size_t end_rblk_num = tt_arg->r_max / tt_arg->blk_rows;
    const size_t start_cblk_num = tt_arg->c_min / tt_arg->blk_cols;
    const size_t end_cblk_num = tt_arg->c_max / tt_arg->blk_cols;
    size_t rblk_num, rblk_min, rblk_max, cblk_num, cblk_min, cblk_max;

    for (rblk_num = start_rblk_num; rblk_num < end_rblk_num; rblk_num++) {
        rblk_min = rblk_num * tt_arg->blk_rows;
        rblk_max = rblk_min + tt_arg->blk_rows;
        for (cblk_num = start_cblk_num; cblk_num < end_cblk_num; cblk_num++) {
            cblk_min = cblk_num * tt_arg->blk_cols;
            cblk_max = cblk_min + tt_arg->blk_cols;
            TRANSPOSE_BLK((const float* restrict)tt_arg->A,
                          (float* restrict)tt_arg->B,
                          tt_arg->A_rows, tt_arg->A_cols,
                          rblk_min, cblk_min, rblk_max, cblk_max);
        }
    }

    pthread_exit((void *)tt_arg->thr_num);
}

static void *transpose_thread_blocked_dbl(void *args) {
    const struct tr_thread_arg *tt_arg = (const struct tr_thread_arg *)args;
    const size_t start_rblk_num = tt_arg->r_min / tt_arg->blk_rows;
    const size_t end_rblk_num = tt_arg->r_max / tt_arg->blk_rows;
    const size_t start_cblk_num = tt_arg->c_min / tt_arg->blk_cols;
    const size_t end_cblk_num = tt_arg->c_max / tt_arg->blk_cols;
    size_t rblk_num, rblk_min, rblk_max, cblk_num, cblk_min, cblk_max;

    for (rblk_num = start_rblk_num; rblk_num < end_rblk_num; rblk_num++) {
        rblk_min = rblk_num * tt_arg->blk_rows;
        rblk_max = rblk_min + tt_arg->blk_rows;
        for (cblk_num = start_cblk_num; cblk_num < end_cblk_num; cblk_num++) {
            cblk_min = cblk_num * tt_arg->blk_cols;
            cblk_max = cblk_min + tt_arg->blk_cols;
            TRANSPOSE_BLK((const double* restrict)tt_arg->A,
                          (double* restrict)tt_arg->B,
                          tt_arg->A_rows, tt_arg->A_cols,
                          rblk_min, cblk_min, rblk_max, cblk_max);
        }
    }

    pthread_exit((void *)tt_arg->thr_num);
}

static void *transpose_thread_blocked_flt_cmplx(void *args) {
    const struct tr_thread_arg *tt_arg = (const struct tr_thread_arg *)args;
    const size_t start_rblk_num = tt_arg->r_min / tt_arg->blk_rows;
    const size_t end_rblk_num = tt_arg->r_max / tt_arg->blk_rows;
    const size_t start_cblk_num = tt_arg->c_min / tt_arg->blk_cols;
    const size_t end_cblk_num = tt_arg->c_max / tt_arg->blk_cols;
    size_t rblk_num, rblk_min, rblk_max, cblk_num, cblk_min, cblk_max;

    for (rblk_num = start_rblk_num; rblk_num < end_rblk_num; rblk_num++) {
        rblk_min = rblk_num * tt_arg->blk_rows;
        rblk_max = rblk_min + tt_arg->blk_rows;
        for (cblk_num = start_cblk_num; cblk_num < end_cblk_num; cblk_num++) {
            cblk_min = cblk_num * tt_arg->blk_cols;
            cblk_max = cblk_min + tt_arg->blk_cols;
            TRANSPOSE_BLK((const float complex* restrict)tt_arg->A,
                          (float complex* restrict)tt_arg->B,
                          tt_arg->A_rows, tt_arg->A_cols,
                          rblk_min, cblk_min, rblk_max, cblk_max);
        }
    }

    pthread_exit((void *)tt_arg->thr_num);
}

static void *transpose_thread_blocked_dbl_cmplx(void *args) {
    const struct tr_thread_arg *tt_arg = (const struct tr_thread_arg *)args;
    const size_t start_rblk_num = tt_arg->r_min / tt_arg->blk_rows;
    const size_t end_rblk_num = tt_arg->r_max / tt_arg->blk_rows;
    const size_t start_cblk_num = tt_arg->c_min / tt_arg->blk_cols;
    const size_t end_cblk_num = tt_arg->c_max / tt_arg->blk_cols;
    size_t rblk_num, rblk_min, rblk_max, cblk_num, cblk_min, cblk_max;

    for (rblk_num = start_rblk_num; rblk_num < end_rblk_num; rblk_num++) {
        rblk_min = rblk_num * tt_arg->blk_rows;
        rblk_max = rblk_min + tt_arg->blk_rows;
        for (cblk_num = start_cblk_num; cblk_num < end_cblk_num; cblk_num++) {
            cblk_min = cblk_num * tt_arg->blk_cols;
            cblk_max = cblk_min + tt_arg->blk_cols;
            TRANSPOSE_BLK((const double complex* restrict)tt_arg->A,
                          (double complex* restrict)tt_arg->B,
                          tt_arg->A_rows, tt_arg->A_cols,
                          rblk_min, cblk_min, rblk_max, cblk_max);
        }
    }

    pthread_exit((void *)tt_arg->thr_num);
}

static void transpose_threads_row(const void* restrict A, void* restrict B,
                                  size_t A_rows, size_t A_cols, size_t num_thr,
                                  void *(*start_routine)(void *))
{
    size_t thr_num, r_min, r_max;
    pthread_t *threads = assert_malloc(num_thr * sizeof(pthread_t));
    struct tr_thread_arg *args = assert_malloc(num_thr * sizeof(struct tr_thread_arg));
    // divide the rows as evenly as possible among the threads
    const size_t num_thr_with_max_rows = A_rows % num_thr;
    const size_t min_rows_per_thread = A_rows / num_thr;
    const size_t max_rows_per_thread = min_rows_per_thread + 1;

    for (thr_num = 0; thr_num < num_thr; thr_num++) {
        if (thr_num < num_thr_with_max_rows) {
            r_min = thr_num * max_rows_per_thread;
            r_max = r_min + max_rows_per_thread;
        } else {
            r_min = num_thr_with_max_rows * max_rows_per_thread +
                    (thr_num - num_thr_with_max_rows) * min_rows_per_thread;
            r_max = r_min + min_rows_per_thread;
        }
        tt_arg_init(&args[thr_num], A, B, A_rows, A_cols,
                    r_min, r_max, 0, A_cols, 0, 0, thr_num);
        errno = pthread_create(&threads[thr_num], NULL, start_routine,
                               &args[thr_num]);
        if (errno) {
            perror("pthread_create");
            exit(errno);
        }
    }

    // wait for the other threads
    for (thr_num = 0; thr_num < num_thr; thr_num++) {
        errno = pthread_join(threads[thr_num], NULL);
        if (errno) {
            perror("pthread_join");
            exit(errno);
        }
    }

    free(args);
    free(threads);
}

static void transpose_threads_col(const void* restrict A, void* restrict B,
                                  size_t A_rows, size_t A_cols, size_t num_thr,
                                  void *(*start_routine)(void *))
{
    size_t thr_num, c_min, c_max;
    pthread_t *threads = assert_malloc(num_thr * sizeof(pthread_t));
    struct tr_thread_arg *args = assert_malloc(num_thr * sizeof(struct tr_thread_arg));
    // divide the columns as evenly as possible among the threads
    const size_t num_thr_with_max_cols = A_cols % num_thr;
    const size_t min_cols_per_thread = A_cols / num_thr;
    const size_t max_cols_per_thread = min_cols_per_thread + 1;

    for (thr_num = 0; thr_num < num_thr; thr_num++) {
        if (thr_num < num_thr_with_max_cols) {
            c_min = thr_num * max_cols_per_thread;
            c_max = c_min + max_cols_per_thread;
        } else {
            c_min = num_thr_with_max_cols * max_cols_per_thread +
                    (thr_num - num_thr_with_max_cols) * min_cols_per_thread;
            c_max = c_min + min_cols_per_thread;
        }
        tt_arg_init(&args[thr_num], A, B, A_rows, A_cols,
                    0, A_rows, c_min, c_max, 0, 0, thr_num);
        errno = pthread_create(&threads[thr_num], NULL, start_routine,
                               &args[thr_num]);
        if (errno) {
            perror("pthread_create");
            exit(errno);
        }
    }

    // wait for the other threads
    for (thr_num = 0; thr_num < num_thr; thr_num++) {
        errno = pthread_join(threads[thr_num], NULL);
        if (errno) {
            perror("pthread_join");
            exit(errno);
        }
    }

    free(args);
    free(threads);
}

void transpose_flt_threads_row(const float* restrict A, float* restrict B,
                               size_t A_rows, size_t A_cols,
                               size_t num_thr)
{
    transpose_threads_row(A, B, A_rows, A_cols, num_thr, &transpose_thread_flt);
}

void transpose_dbl_threads_row(const double* restrict A, double* restrict B,
                               size_t A_rows, size_t A_cols,
                               size_t num_thr)
{
    transpose_threads_row(A, B, A_rows, A_cols, num_thr, &transpose_thread_dbl);
}

void transpose_flt_threads_col(const float* restrict A, float* restrict B,
                               size_t A_rows, size_t A_cols,
                               size_t num_thr)
{
    transpose_threads_col(A, B, A_rows, A_cols, num_thr, &transpose_thread_flt);
}

void transpose_dbl_threads_col(const double* restrict A, double* restrict B,
                               size_t A_rows, size_t A_cols,
                               size_t num_thr)
{
    transpose_threads_col(A, B, A_rows, A_cols, num_thr, &transpose_thread_dbl);
}

void transpose_flt_cmplx_threads_row(const float complex* restrict A,
                                     float complex* restrict B,
                                     size_t A_rows, size_t A_cols,
                                     size_t num_thr)
{
    transpose_threads_row(A, B, A_rows, A_cols, num_thr,
                          &transpose_thread_flt_cmplx);
}

void transpose_dbl_cmplx_threads_row(const double complex* restrict A,
                                     double complex* restrict B,
                                     size_t A_rows, size_t A_cols,
                                     size_t num_thr)
{
    transpose_threads_row(A, B, A_rows, A_cols, num_thr,
                          &transpose_thread_dbl_cmplx);
}

void transpose_flt_cmplx_threads_col(const float complex* restrict A,
                                     float complex* restrict B,
                                     size_t A_rows, size_t A_cols,
                                     size_t num_thr)
{
    transpose_threads_col(A, B, A_rows, A_cols, num_thr,
                          &transpose_thread_flt_cmplx);
}

void transpose_dbl_cmplx_threads_col(const double complex* restrict A,
                                     double complex* restrict B,
                                     size_t A_rows, size_t A_cols,
                                     size_t num_thr)
{
    transpose_threads_col(A, B, A_rows, A_cols, num_thr,
                          &transpose_thread_dbl_cmplx);
}

static void transpose_threads_row_blocked(const void* restrict A,
                                          void* restrict B,
                                          size_t A_rows, size_t A_cols,
                                          size_t num_thr,
                                          size_t blk_rows, size_t blk_cols,
                                          void *(*start_routine)(void *))
{
    size_t r_min, r_max, thr_num;
    const size_t rows_per_thr = A_rows / num_thr;
    pthread_t *threads = assert_malloc(num_thr * sizeof(pthread_t));
    struct tr_thread_arg *args = assert_malloc(num_thr * sizeof(struct tr_thread_arg));
    assert(A_rows % num_thr == 0);

    for (thr_num = 0; thr_num < num_thr; thr_num++) {
        r_min = thr_num * rows_per_thr;
        r_max = r_min + rows_per_thr;
        tt_arg_init(&args[thr_num], A, B, A_rows, A_cols,
                    r_min, r_max, 0, A_cols, blk_rows, blk_cols, thr_num);
        errno = pthread_create(&threads[thr_num], NULL, start_routine,
                               &args[thr_num]);
        if (errno) {
            perror("pthread_create");
            exit(errno);
        }
    }

    // wait for the other threads
    for (thr_num = 0; thr_num < num_thr; thr_num++) {
        errno = pthread_join(threads[thr_num], NULL);
        if (errno) {
            perror("pthread_join");
            exit(errno);
        }
    }

    free(args);
    free(threads);
}

void transpose_threads_col_blocked(const void* restrict A,
                                   void* restrict B,
                                   size_t A_rows, size_t A_cols,
                                   size_t num_thr,
                                   size_t blk_rows, size_t blk_cols,
                                   void *(*start_routine)(void *))
{
    size_t c_min, c_max, thr_num;
    const size_t cols_per_thr = A_cols / num_thr;
    pthread_t *threads = assert_malloc(num_thr * sizeof(pthread_t));
    struct tr_thread_arg *args = assert_malloc(num_thr * sizeof(struct tr_thread_arg));
    assert(A_cols % num_thr == 0);

    for (thr_num = 0; thr_num < num_thr; thr_num++) {
        c_min = thr_num * cols_per_thr;
        c_max = c_min + cols_per_thr;
        tt_arg_init(&args[thr_num], A, B, A_rows, A_cols,
                    0, A_rows, c_min, c_max, blk_rows, blk_cols, thr_num);
        errno = pthread_create(&threads[thr_num], NULL, start_routine,
                               &args[thr_num]);
        if (errno) {
            perror("pthread_create");
            exit(errno);
        }
    }

    // wait for the other threads
    for (thr_num = 0; thr_num < num_thr; thr_num++) {
        errno = pthread_join(threads[thr_num], NULL);
        if (errno) {
            perror("pthread_join");
            exit(errno);
        }
    }

    free(args);
    free(threads);
}

void transpose_flt_threads_row_blocked(const float* restrict A,
                                       float* restrict B,
                                       size_t A_rows, size_t A_cols,
                                       size_t num_thr,
                                       size_t blk_rows, size_t blk_cols)
{
    transpose_threads_row_blocked(A, B, A_rows, A_cols, num_thr, blk_rows,
                                  blk_cols, &transpose_thread_blocked_flt);
}

void transpose_dbl_threads_row_blocked(const double* restrict A,
                                       double* restrict B,
                                       size_t A_rows, size_t A_cols,
                                       size_t num_thr,
                                       size_t blk_rows, size_t blk_cols)
{
    transpose_threads_row_blocked(A, B, A_rows, A_cols, num_thr, blk_rows,
                                  blk_cols, &transpose_thread_blocked_dbl);
}

void transpose_flt_threads_col_blocked(const float* restrict A,
                                       float* restrict B,
                                       size_t A_rows, size_t A_cols,
                                       size_t num_thr,
                                       size_t blk_rows, size_t blk_cols)
{
    transpose_threads_col_blocked(A, B, A_rows, A_cols, num_thr, blk_rows,
                                  blk_cols, &transpose_thread_blocked_flt);
}

void transpose_dbl_threads_col_blocked(const double* restrict A,
                                       double* restrict B,
                                       size_t A_rows, size_t A_cols,
                                       size_t num_thr,
                                       size_t blk_rows, size_t blk_cols)
{
    transpose_threads_col_blocked(A, B, A_rows, A_cols, num_thr, blk_rows,
                                  blk_cols, &transpose_thread_blocked_dbl);
}

void transpose_flt_cmplx_threads_row_blocked(const float complex* restrict A,
                                             float complex* restrict B,
                                             size_t A_rows, size_t A_cols,
                                             size_t num_thr,
                                             size_t blk_rows, size_t blk_cols)
{
    transpose_threads_row_blocked(A, B, A_rows, A_cols, num_thr, blk_rows,
                                  blk_cols, &transpose_thread_blocked_flt_cmplx);
}

void transpose_dbl_cmplx_threads_row_blocked(const double complex* restrict A,
                                             double complex* restrict B,
                                             size_t A_rows, size_t A_cols,
                                             size_t num_thr,
                                             size_t blk_rows, size_t blk_cols)
{
    transpose_threads_row_blocked(A, B, A_rows, A_cols, num_thr, blk_rows,
                                  blk_cols, &transpose_thread_blocked_dbl_cmplx);
}

void transpose_flt_cmplx_threads_col_blocked(const float complex* restrict A,
                                             float complex* restrict B,
                                             size_t A_rows, size_t A_cols,
                                             size_t num_thr,
                                             size_t blk_rows, size_t blk_cols)
{
    transpose_threads_col_blocked(A, B, A_rows, A_cols, num_thr, blk_rows,
                                  blk_cols, &transpose_thread_blocked_flt_cmplx);
}

void transpose_dbl_cmplx_threads_col_blocked(const double complex* restrict A,
                                             double complex* restrict B,
                                             size_t A_rows, size_t A_cols,
                                             size_t num_thr,
                                             size_t blk_rows, size_t blk_cols)
{
    transpose_threads_col_blocked(A, B, A_rows, A_cols, num_thr, blk_rows,
                                  blk_cols, &transpose_thread_blocked_dbl_cmplx);
}
