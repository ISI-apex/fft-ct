/**
 * Transpose functions.
 *
 * @author Kaushik Datta <kdatta@isi.edu>
 * @author Connor Imes <cimes@isi.edu>
 * @date 2019-08-06
 */
#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>

#include "transpose-threads.h"
#include "util.h"

struct tr_thread_arg {
    const void* restrict A;
    void* restrict B;
    size_t A_rows, A_cols, min, max, thr_num;
};

static void tt_arg_init(struct tr_thread_arg *tt_arg,
                        const void* restrict A, void* restrict B,
                        size_t A_rows, size_t A_cols,
                        size_t min, size_t max, size_t thr_num)
{
    tt_arg->A = A;
    tt_arg->B = B;
    tt_arg->A_rows = A_rows;
    tt_arg->A_cols = A_cols;
    tt_arg->min = min;
    tt_arg->max = max;
    tt_arg->thr_num = thr_num;
}

static void *transpose_flt_thread_row(void *args) {
    struct tr_thread_arg *tt_arg = (struct tr_thread_arg *)args;
    const float* restrict A = (float *)tt_arg->A;
    float* restrict B = (float *)tt_arg->B;
    size_t r, c;

    for (r = tt_arg->min; r < tt_arg->max; r++) {
        for (c = 0; c < tt_arg->A_cols; c++) {
            B[c * tt_arg->A_rows + r] = A[r * tt_arg->A_cols + c];
        }
    }

    pthread_exit((void*) tt_arg->thr_num);
}

void transpose_flt_threads_row(const float* restrict A, float* restrict B,
                               size_t A_rows, size_t A_cols,
                               size_t num_thr) {
    pthread_t *threads;
    pthread_attr_t attr;
    struct tr_thread_arg *args, *tt_arg;
    size_t thr_num, r_min, r_max;
    size_t num_thr_with_max_rows, min_rows_per_thread, max_rows_per_thread;
    int rc;
    void *status;

    threads = assert_malloc(num_thr * sizeof(pthread_t));
    // initialize and set thread detached attribute
    pthread_attr_init(&attr);
    pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);

    args = assert_malloc(num_thr * sizeof(struct tr_thread_arg));

    // divide the rows as evenly as possible among the threads
    num_thr_with_max_rows = A_rows % num_thr;
    min_rows_per_thread = A_rows / num_thr;
    max_rows_per_thread = min_rows_per_thread + 1;

    for (thr_num=0; thr_num<num_thr; thr_num++) {
        if (thr_num < num_thr_with_max_rows) {
            r_min = thr_num * max_rows_per_thread;
            r_max = r_min + max_rows_per_thread;
        } else {
            r_min = num_thr_with_max_rows * max_rows_per_thread + (thr_num - num_thr_with_max_rows) * min_rows_per_thread;
            r_max = r_min + min_rows_per_thread;
        }

        tt_arg = args + thr_num;
        tt_arg_init(tt_arg, A, B, A_rows, A_cols, r_min, r_max, thr_num);

        rc = pthread_create(&threads[thr_num], &attr, &transpose_flt_thread_row, (void *)tt_arg);
        if (rc) {
            fprintf(stderr, "ERROR: return code from pthread_create() is %d\n", rc);
            exit(-1);
        }
    }

    // free attribute and wait for the other threads
    pthread_attr_destroy(&attr);
    for(thr_num=0; thr_num<num_thr; thr_num++) {
        rc = pthread_join(threads[thr_num], &status);
        if (rc) {
            fprintf(stderr, "ERROR: return code from pthread_join() is %d\n", rc);
            exit(-1);
        }
    }

    free(args);
    free(threads);
}

static void *transpose_dbl_thread_row(void *args) {
    struct tr_thread_arg *tt_arg = (struct tr_thread_arg *)args;
    const double* restrict A = (double *)tt_arg->A;
    double* restrict B = (double *)tt_arg->B;
    size_t r, c;

    for (r = tt_arg->min; r < tt_arg->max; r++) {
        for (c = 0; c < tt_arg->A_cols; c++) {
            B[c * tt_arg->A_rows + r] = A[r * tt_arg->A_cols + c];
        }
    }

    pthread_exit((void*) tt_arg->thr_num);
}

void transpose_dbl_threads_row(const double* restrict A, double* restrict B,
                               size_t A_rows, size_t A_cols,
                               size_t num_thr) {
    pthread_t *threads;
    pthread_attr_t attr;
    struct tr_thread_arg *args, *tt_arg;
    size_t thr_num, r_min, r_max;
    size_t num_thr_with_max_rows, min_rows_per_thread, max_rows_per_thread;
    int rc;
    void *status;

    threads = assert_malloc(num_thr * sizeof(pthread_t));
    // initialize and set thread detached attribute
    pthread_attr_init(&attr);
    pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);

    args = assert_malloc(num_thr * sizeof(struct tr_thread_arg));

    // divide the rows as evenly as possible among the threads
    num_thr_with_max_rows = A_rows % num_thr;
    min_rows_per_thread = A_rows / num_thr;
    max_rows_per_thread = min_rows_per_thread + 1;

    for (thr_num=0; thr_num<num_thr; thr_num++) {
        if (thr_num < num_thr_with_max_rows) {
            r_min = thr_num * max_rows_per_thread;
            r_max = r_min + max_rows_per_thread;
        } else {
            r_min = num_thr_with_max_rows * max_rows_per_thread + (thr_num - num_thr_with_max_rows) * min_rows_per_thread;
            r_max = r_min + min_rows_per_thread;
        }

        tt_arg = args + thr_num;
        tt_arg_init(tt_arg, A, B, A_rows, A_cols, r_min, r_max, thr_num);

        rc = pthread_create(&threads[thr_num], &attr, &transpose_dbl_thread_row, (void *)tt_arg);
        if (rc) {
            fprintf(stderr, "ERROR: return code from pthread_create() is %d\n", rc);
            exit(-1);
        }
    }

    // free attribute and wait for the other threads
    pthread_attr_destroy(&attr);
    for(thr_num=0; thr_num<num_thr; thr_num++) {
        rc = pthread_join(threads[thr_num], &status);
        if (rc) {
            fprintf(stderr, "ERROR: return code from pthread_join() is %d\n", rc);
            exit(-1);
        }
    }

    free(args);
    free(threads);
}

static void *transpose_flt_thread_col(void *args) {
    struct tr_thread_arg *tt_arg = (struct tr_thread_arg *)args;
    const float* restrict A = (float *)tt_arg->A;
    float* restrict B = (float *)tt_arg->B;
    size_t r, c;

    for (r = 0; r < tt_arg->A_rows; r++) {
        for (c = tt_arg->min; c < tt_arg->max; c++) {
            B[c * tt_arg->A_rows + r] = A[r * tt_arg->A_cols + c];
        }
    }

    pthread_exit((void*) tt_arg->thr_num);
}

void transpose_flt_threads_col(const float* restrict A, float* restrict B,
                               size_t A_rows, size_t A_cols,
                               size_t num_thr) {
    pthread_t *threads;
    pthread_attr_t attr;
    struct tr_thread_arg *args, *tt_arg;
    size_t thr_num, c_min, c_max;
    size_t num_thr_with_max_cols, min_cols_per_thread, max_cols_per_thread;
    int rc;
    void *status;

    threads = assert_malloc(num_thr * sizeof(pthread_t));
    // initialize and set thread detached attribute
    pthread_attr_init(&attr);
    pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);

    args = assert_malloc(num_thr * sizeof(struct tr_thread_arg));

    // divide the columns as evenly as possible among the threads
    num_thr_with_max_cols = A_cols % num_thr;
    min_cols_per_thread = A_cols / num_thr;
    max_cols_per_thread = min_cols_per_thread + 1;

    for (thr_num=0; thr_num<num_thr; thr_num++) {
        if (thr_num < num_thr_with_max_cols) {
            c_min = thr_num * max_cols_per_thread;
            c_max = c_min + max_cols_per_thread;
        } else {
            c_min = num_thr_with_max_cols * max_cols_per_thread + (thr_num - num_thr_with_max_cols) * min_cols_per_thread;
            c_max = c_min + min_cols_per_thread;
        }

        tt_arg = args + thr_num;
        tt_arg_init(tt_arg, A, B, A_rows, A_cols, c_min, c_max, thr_num);

        rc = pthread_create(&threads[thr_num], &attr, &transpose_flt_thread_col, (void *)tt_arg);
        if (rc) {
            fprintf(stderr, "ERROR: return code from pthread_create() is %d\n", rc);
            exit(-1);
        }
    }

    // free attribute and wait for the other threads
    pthread_attr_destroy(&attr);
    for(thr_num=0; thr_num<num_thr; thr_num++) {
        rc = pthread_join(threads[thr_num], &status);
        if (rc) {
            fprintf(stderr, "ERROR: return code from pthread_join() is %d\n", rc);
            exit(-1);
        }
    }

    free(args);
    free(threads);
}

static void *transpose_dbl_thread_col(void *args) {
    struct tr_thread_arg *tt_arg = (struct tr_thread_arg *)args;
    const double* restrict A = (double *)tt_arg->A;
    double* restrict B = (double *)tt_arg->B;
    size_t r, c;

    for (r = 0; r < tt_arg->A_rows; r++) {
        for (c = tt_arg->min; c < tt_arg->max; c++) {
            B[c * tt_arg->A_rows + r] = A[r * tt_arg->A_cols + c];
        }
    }

    pthread_exit((void*) tt_arg->thr_num);
}

void transpose_dbl_threads_col(const double* restrict A, double* restrict B,
                               size_t A_rows, size_t A_cols,
                               size_t num_thr) {
    pthread_t *threads;
    pthread_attr_t attr;
    struct tr_thread_arg *args, *tt_arg;
    size_t thr_num, c_min, c_max;
    size_t num_thr_with_max_cols, min_cols_per_thread, max_cols_per_thread;
    int rc;
    void *status;

    threads = assert_malloc(num_thr * sizeof(pthread_t));
    // initialize and set thread detached attribute
    pthread_attr_init(&attr);
    pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);

    args = assert_malloc(num_thr * sizeof(struct tr_thread_arg));

    // divide the columns as evenly as possible among the threads
    num_thr_with_max_cols = A_cols % num_thr;
    min_cols_per_thread = A_cols / num_thr;
    max_cols_per_thread = min_cols_per_thread + 1;

    for (thr_num=0; thr_num<num_thr; thr_num++) {
        if (thr_num < num_thr_with_max_cols) {
            c_min = thr_num * max_cols_per_thread;
            c_max = c_min + max_cols_per_thread;
        } else {
            c_min = num_thr_with_max_cols * max_cols_per_thread + (thr_num - num_thr_with_max_cols) * min_cols_per_thread;
            c_max = c_min + min_cols_per_thread;
        }

        tt_arg = args + thr_num;
        tt_arg_init(tt_arg, A, B, A_rows, A_cols, c_min, c_max, thr_num);

        rc = pthread_create(&threads[thr_num], &attr, &transpose_dbl_thread_col, (void *)tt_arg);
        if (rc) {
            fprintf(stderr, "ERROR: return code from pthread_create() is %d\n", rc);
            exit(-1);
        }
    }

    // free attribute and wait for the other threads
    pthread_attr_destroy(&attr);
    for(thr_num=0; thr_num<num_thr; thr_num++) {
        rc = pthread_join(threads[thr_num], &status);
        if (rc) {
            fprintf(stderr, "ERROR: return code from pthread_join() is %d\n", rc);
            exit(-1);
        }
    }

    free(args);
    free(threads);
}
