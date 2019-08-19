/**
 * Transpose functions.
 *
 * @author Kaushik Datta <kdatta@isi.edu>
 * @date 2019-08-15
 */
#include <assert.h>
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>

// intrinsics
#include <immintrin.h>

#include "transpose-threads-avx.h"
#include "util.h"

struct tr_thread_arg {
    const void* restrict A;
    void* restrict B;
    size_t A_rows, A_cols, r_min, r_max, c_min, c_max, thr_num;
};

static void tt_arg_init(struct tr_thread_arg *tt_arg,
                        const void* restrict A, void* restrict B,
                        size_t A_rows, size_t A_cols,
                        size_t r_min, size_t r_max, size_t c_min, size_t c_max,
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
    tt_arg->thr_num = thr_num;
}

static void *transpose_thread_blocked_dbl(void *args) {
    // used for swapping 2x2 blocks using _mm512_permutex2var_pd()
    static const __m512i idx_2x2_0 = {
        0x0000, 0x0001, 0x0008, 0x0009, 0x0004, 0x0005, 0x000c, 0x000d
    };
    static const __m512i idx_2x2_1 = {
        0x000a, 0x000b, 0x0002, 0x0003, 0x000e, 0x000f, 0x0006, 0x0007
    };
    // used for swapping 4x4 blocks using _mm512_permutex2var_pd()
    static const __m512i idx_4x4_0 = {
        0x0000, 0x0001, 0x0002, 0x0003, 0x0008, 0x0009, 0x000a, 0x000b
    };
    static const __m512i idx_4x4_1 = {
        0x000c, 0x000d, 0x000e, 0x000f, 0x0004, 0x0005, 0x0006, 0x0007
    };
    const struct tr_thread_arg *tt_arg = (struct tr_thread_arg *)args;
    const double* restrict A = tt_arg->A;
    double* restrict B = tt_arg->B;
    const size_t start_rblk_num = tt_arg->r_min / 8;
    const size_t end_rblk_num = tt_arg->r_max / 8;
    const size_t start_cblk_num = tt_arg->c_min / 8;
    const size_t end_cblk_num = tt_arg->c_max / 8;
    const double *A_block;
    double *B_block;
    size_t rblk_num, cblk_num, r_min, c_min;
    // alternate the reads and writes between the r and s vector registers, all
    // of which hold matrix rows
    __m512d r[8], s[8];

    assert(tt_arg->A_rows % 8 == 0);
    assert(tt_arg->A_cols % 8 == 0);
    assert(tt_arg->r_min % 8 == 0);
    assert(tt_arg->r_max % 8 == 0);
    assert(tt_arg->c_min % 8 == 0);
    assert(tt_arg->c_max % 8 == 0);

    for (rblk_num = start_rblk_num; rblk_num < end_rblk_num; rblk_num++) {
        r_min = rblk_num * 8;
        for (cblk_num = start_cblk_num; cblk_num < end_cblk_num; cblk_num++) {
            c_min = cblk_num * 8;

            A_block = &A[r_min * tt_arg->A_cols + c_min];
            B_block = &B[c_min * tt_arg->A_rows + r_min];

            // read 8x8 block of read array
            r[0] = _mm512_load_pd(&A_block[0]);
            r[1] = _mm512_load_pd(&A_block[tt_arg->A_cols]);
            r[2] = _mm512_load_pd(&A_block[2*tt_arg->A_cols]);
            r[3] = _mm512_load_pd(&A_block[3*tt_arg->A_cols]);
            r[4] = _mm512_load_pd(&A_block[4*tt_arg->A_cols]);
            r[5] = _mm512_load_pd(&A_block[5*tt_arg->A_cols]);
            r[6] = _mm512_load_pd(&A_block[6*tt_arg->A_cols]);
            r[7] = _mm512_load_pd(&A_block[7*tt_arg->A_cols]);

            // shuffle doubles within 128-bit lanes
            s[0] = _mm512_unpacklo_pd(r[0], r[1]);
            s[1] = _mm512_unpackhi_pd(r[0], r[1]);
            s[2] = _mm512_unpacklo_pd(r[2], r[3]);
            s[3] = _mm512_unpackhi_pd(r[2], r[3]);
            s[4] = _mm512_unpacklo_pd(r[4], r[5]);
            s[5] = _mm512_unpackhi_pd(r[4], r[5]);
            s[6] = _mm512_unpacklo_pd(r[6], r[7]);
            s[7] = _mm512_unpackhi_pd(r[6], r[7]);

            // shuffle 2x2 blocks of doubles
            r[0] = _mm512_permutex2var_pd(s[0], idx_2x2_0, s[2]);
            r[1] = _mm512_permutex2var_pd(s[1], idx_2x2_0, s[3]);
            r[2] = _mm512_permutex2var_pd(s[2], idx_2x2_1, s[0]);
            r[3] = _mm512_permutex2var_pd(s[3], idx_2x2_1, s[1]);
            r[4] = _mm512_permutex2var_pd(s[4], idx_2x2_0, s[6]);
            r[5] = _mm512_permutex2var_pd(s[5], idx_2x2_0, s[7]);
            r[6] = _mm512_permutex2var_pd(s[6], idx_2x2_1, s[4]);
            r[7] = _mm512_permutex2var_pd(s[7], idx_2x2_1, s[5]);

            // shuffle 4x4 blocks of doubles
            s[0] = _mm512_permutex2var_pd(r[0], idx_4x4_0, r[4]);
            s[1] = _mm512_permutex2var_pd(r[1], idx_4x4_0, r[5]);
            s[2] = _mm512_permutex2var_pd(r[2], idx_4x4_0, r[6]);
            s[3] = _mm512_permutex2var_pd(r[3], idx_4x4_0, r[7]);
            s[4] = _mm512_permutex2var_pd(r[4], idx_4x4_1, r[0]);
            s[5] = _mm512_permutex2var_pd(r[5], idx_4x4_1, r[1]);
            s[6] = _mm512_permutex2var_pd(r[6], idx_4x4_1, r[2]);
            s[7] = _mm512_permutex2var_pd(r[7], idx_4x4_1, r[3]);

            // write back 8x8 block of write array
#if defined(USE_AVX_STREAMING_STORES)
            _mm512_stream_pd(&B_block[0], s[0]);
            _mm512_stream_pd(&B_block[tt_arg->A_rows], s[1]);
            _mm512_stream_pd(&B_block[2*tt_arg->A_rows], s[2]);
            _mm512_stream_pd(&B_block[3*tt_arg->A_rows], s[3]);
            _mm512_stream_pd(&B_block[4*tt_arg->A_rows], s[4]);
            _mm512_stream_pd(&B_block[5*tt_arg->A_rows], s[5]);
            _mm512_stream_pd(&B_block[6*tt_arg->A_rows], s[6]);
            _mm512_stream_pd(&B_block[7*tt_arg->A_rows], s[7]);
#else
            _mm512_store_pd(&B_block[0], s[0]);
            _mm512_store_pd(&B_block[tt_arg->A_rows], s[1]);
            _mm512_store_pd(&B_block[2*tt_arg->A_rows], s[2]);
            _mm512_store_pd(&B_block[3*tt_arg->A_rows], s[3]);
            _mm512_store_pd(&B_block[4*tt_arg->A_rows], s[4]);
            _mm512_store_pd(&B_block[5*tt_arg->A_rows], s[5]);
            _mm512_store_pd(&B_block[6*tt_arg->A_rows], s[6]);
            _mm512_store_pd(&B_block[7*tt_arg->A_rows], s[7]);
#endif
        }
    }

    pthread_exit((void *)tt_arg->thr_num);
}

void transpose_dbl_threads_avx_intr_8x8_row(const double* restrict A,
                                            double* restrict B,
                                            size_t A_rows, size_t A_cols,
                                            size_t num_thr)
{
    size_t r_min, r_max, thr_num;
    const size_t rows_per_thr = A_rows / num_thr;
    pthread_t *threads = assert_malloc(num_thr * sizeof(pthread_t));
    struct tr_thread_arg *args = assert_malloc(num_thr * sizeof(struct tr_thread_arg));

    assert(rows_per_thr % num_thr == 0);

    for (thr_num = 0; thr_num < num_thr; thr_num++) {
        r_min = thr_num * rows_per_thr;
        r_max = r_min + rows_per_thr;

        tt_arg_init(&args[thr_num], A, B, A_rows, A_cols,
                    r_min, r_max, 0, A_cols, thr_num);
        errno = pthread_create(&threads[thr_num], NULL, &transpose_thread_blocked_dbl,
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

void transpose_dbl_threads_avx_intr_8x8_col(const double* restrict A,
                                            double* restrict B,
                                            size_t A_rows, size_t A_cols,
                                            size_t num_thr)
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
                    0, A_rows, c_min, c_max, thr_num);
        errno = pthread_create(&threads[thr_num], NULL, &transpose_thread_blocked_dbl,
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
