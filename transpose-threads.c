/**
 * Transpose functions.
 *
 * @author Kaushik Datta <kdatta@isi.edu>
 * @date 2019-08-06
 */
#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>

#include "transpose-threads.h"

typedef struct floatRowArg {
    const float* restrict A;
    float* restrict B;
    size_t A_rows, A_cols, r_min, r_max, thr_num;
} fltRowArg;

typedef struct doubleRowArg {
    const double* restrict A;
    double* restrict B;
    size_t A_rows, A_cols, r_min, r_max, thr_num;
} dblRowArg;

typedef struct floatColArg {
    const float* restrict A;
    float* restrict B;
    size_t A_rows, A_cols, c_min, c_max, thr_num;
} fltColArg;

typedef struct doubleColArg {
    const double* restrict A;
    double* restrict B;
    size_t A_rows, A_cols, c_min, c_max, thr_num;
} dblColArg;

void *transpose_flt_thread_row(void *args) {
    const float* restrict A;
    float* restrict B;
    size_t A_rows, A_cols, r, c, r_min, r_max;
    size_t thr_num;

    fltRowArg *myArg = (fltRowArg *)args;
    A = (float *)(myArg->A);
    B = (float *)(myArg->B);
    A_rows = (size_t)(myArg->A_rows);
    A_cols = (size_t)(myArg->A_cols);
    r_min = (size_t)(myArg->r_min);
    r_max = (size_t)(myArg->r_max);
    thr_num = (size_t)(myArg->thr_num);

    for (r = r_min; r < r_max; r++) {
        for (c = 0; c < A_cols; c++) {
            B[c * A_rows + r] = A[r * A_cols + c];
        }
    }

    pthread_exit((void*) thr_num);
}

void transpose_flt_threads_row(const float* restrict A, float* restrict B,
                               size_t A_rows, size_t A_cols,
                               size_t num_thr) {
    pthread_t *threads;
    pthread_attr_t attr;
    fltRowArg *args, *myArg;
    size_t thr_num, r_min, r_max;
    size_t num_thr_with_max_rows, min_rows_per_thread, max_rows_per_thread;
    int rc;
    void *status;

    threads = malloc(num_thr * sizeof(pthread_t));
    // initialize and set thread detached attribute
    pthread_attr_init(&attr);
    pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);

    args = malloc(num_thr * sizeof(fltRowArg));

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

        myArg = args + thr_num;
        myArg->A = A;
        myArg->B = B;
        myArg->A_rows = A_rows;
        myArg->A_cols = A_cols;
        myArg->r_min = r_min;
        myArg->r_max = r_max;
        myArg->thr_num = thr_num;

        rc = pthread_create(&threads[thr_num], &attr, &transpose_flt_thread_row, (void *)myArg);
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

void *transpose_dbl_thread_row(void *args) {
    const double* restrict A;
    double* restrict B;
    size_t A_rows, A_cols, r, c, r_min, r_max;
    size_t thr_num;

    dblRowArg *myArg = (dblRowArg *)args;
    A = (double *)(myArg->A);
    B = (double *)(myArg->B);
    A_rows = (size_t)(myArg->A_rows);
    A_cols = (size_t)(myArg->A_cols);
    r_min = (size_t)(myArg->r_min);
    r_max = (size_t)(myArg->r_max);
    thr_num = (size_t)(myArg->thr_num);

    for (r = r_min; r < r_max; r++) {
        for (c = 0; c < A_cols; c++) {
            B[c * A_rows + r] = A[r * A_cols + c];
        }
    }

    pthread_exit((void*) thr_num);
}

void transpose_dbl_threads_row(const double* restrict A, double* restrict B,
                               size_t A_rows, size_t A_cols,
                               size_t num_thr) {
    pthread_t *threads;
    pthread_attr_t attr;
    dblRowArg *args, *myArg;
    size_t thr_num, r_min, r_max;
    size_t num_thr_with_max_rows, min_rows_per_thread, max_rows_per_thread;
    int rc;
    void *status;

    threads = malloc(num_thr * sizeof(pthread_t));
    // initialize and set thread detached attribute
    pthread_attr_init(&attr);
    pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);

    args = malloc(num_thr * sizeof(dblRowArg));

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

        myArg = args + thr_num;
        myArg->A = A;
        myArg->B = B;
        myArg->A_rows = A_rows;
        myArg->A_cols = A_cols;
        myArg->r_min = r_min;
        myArg->r_max = r_max;
        myArg->thr_num = thr_num;

        rc = pthread_create(&threads[thr_num], &attr, &transpose_dbl_thread_row, (void *)myArg);
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

void *transpose_flt_thread_col(void *args) {
    const float* restrict A;
    float* restrict B;
    size_t A_rows, A_cols, r, c, c_min, c_max;
    size_t thr_num;

    fltColArg *myArg = (fltColArg *)args;
    A = (float *)(myArg->A);
    B = (float *)(myArg->B);
    A_rows = (size_t)(myArg->A_rows);
    A_cols = (size_t)(myArg->A_cols);
    c_min = (size_t)(myArg->c_min);
    c_max = (size_t)(myArg->c_max);
    thr_num = (size_t)(myArg->thr_num);

    for (r = 0; r < A_rows; r++) {
        for (c = c_min; c < c_max; c++) {
            B[c * A_rows + r] = A[r * A_cols + c];
        }
    }

    pthread_exit((void*) thr_num);
}

void transpose_flt_threads_col(const float* restrict A, float* restrict B,
                               size_t A_rows, size_t A_cols,
                               size_t num_thr) {
    pthread_t *threads;
    pthread_attr_t attr;
    fltColArg *args, *myArg;
    size_t thr_num, c_min, c_max;
    size_t num_thr_with_max_cols, min_cols_per_thread, max_cols_per_thread;
    int rc;
    void *status;

    threads = malloc(num_thr * sizeof(pthread_t));
    // initialize and set thread detached attribute
    pthread_attr_init(&attr);
    pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);

    args = malloc(num_thr * sizeof(fltColArg));

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

        myArg = args + thr_num;
        myArg->A = A;
        myArg->B = B;
        myArg->A_rows = A_rows;
        myArg->A_cols = A_cols;
        myArg->c_min = c_min;
        myArg->c_max = c_max;
        myArg->thr_num = thr_num;

        rc = pthread_create(&threads[thr_num], &attr, &transpose_flt_thread_col, (void *)myArg);
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

void *transpose_dbl_thread_col(void *args) {
    const double* restrict A;
    double* restrict B;
    size_t A_rows, A_cols, r, c, c_min, c_max;
    size_t thr_num;

    dblColArg *myArg = (dblColArg *)args;
    A = (double *)(myArg->A);
    B = (double *)(myArg->B);
    A_rows = (size_t)(myArg->A_rows);
    A_cols = (size_t)(myArg->A_cols);
    c_min = (size_t)(myArg->c_min);
    c_max = (size_t)(myArg->c_max);
    thr_num = (size_t)(myArg->thr_num);

    for (r = 0; r < A_rows; r++) {
        for (c = c_min; c < c_max; c++) {
            B[c * A_rows + r] = A[r * A_cols + c];
        }
    }

    pthread_exit((void*) thr_num);
}

void transpose_dbl_threads_col(const double* restrict A, double* restrict B,
                               size_t A_rows, size_t A_cols,
                               size_t num_thr) {
    pthread_t *threads;
    pthread_attr_t attr;
    dblColArg *args, *myArg;
    size_t thr_num, c_min, c_max;
    size_t num_thr_with_max_cols, min_cols_per_thread, max_cols_per_thread;
    int rc;
    void *status;

    threads = malloc(num_thr * sizeof(pthread_t));
    // initialize and set thread detached attribute
    pthread_attr_init(&attr);
    pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);

    args = malloc(num_thr * sizeof(dblColArg));

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

        myArg = args + thr_num;
        myArg->A = A;
        myArg->B = B;
        myArg->A_rows = A_rows;
        myArg->A_cols = A_cols;
        myArg->c_min = c_min;
        myArg->c_max = c_max;
        myArg->thr_num = thr_num;

        rc = pthread_create(&threads[thr_num], &attr, &transpose_dbl_thread_col, (void *)myArg);
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
