/**
 * Transpose functions.
 *
 * @author Kaushik Datta <kdatta@isi.edu>
 * @author Connor Imes <cimes@isi.edu>
 * @date 2019-08-07
 */
#include <assert.h>
#include <inttypes.h>
#include <stdlib.h>

// intrinsics
#include <immintrin.h>

#include "transpose-avx.h"

/*
 * This function uses intrinsics to transpose an 8x8 block of doubles
 * using a recursive transpose algorithm.  It will not work correctly
 * unless both A_rows and A_cols are multiples of 8.
 */
void transpose_dbl_avx_intr_8x8(const double* restrict A, double* restrict B,
                                size_t A_rows, size_t A_cols)
{
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
    const double *A_block;
    double *B_block;
    size_t i_min, j_min;
    size_t num_row_blocks, num_col_blocks;
    size_t rblk_num, cblk_num;
    // alternate the reads and writes between the r and s vector registers, all
    // of which hold matrix rows
    __m512d r0, r1, r2, r3, r4, r5, r6, r7;
    __m512d s0, s1, s2, s3, s4, s5, s6, s7;

    assert(A_rows % 8 == 0);
    assert(A_cols % 8 == 0);

    num_row_blocks = A_rows / 8;
    num_col_blocks = A_cols / 8;

    // perform transpose over all blocks
    for (rblk_num = 0; rblk_num < num_row_blocks; rblk_num++) {
        i_min = rblk_num * 8;
        for (cblk_num = 0; cblk_num < num_col_blocks; cblk_num++) {
            j_min = cblk_num * 8;

            A_block = &A[i_min * A_cols + j_min];
            B_block = &B[j_min * A_rows + i_min];

            // read 8x8 block of read array
            r0 = _mm512_load_pd(&A_block[0]);
            r1 = _mm512_load_pd(&A_block[A_cols]);
            r2 = _mm512_load_pd(&A_block[2*A_cols]);
            r3 = _mm512_load_pd(&A_block[3*A_cols]);
            r4 = _mm512_load_pd(&A_block[4*A_cols]);
            r5 = _mm512_load_pd(&A_block[5*A_cols]);
            r6 = _mm512_load_pd(&A_block[6*A_cols]);
            r7 = _mm512_load_pd(&A_block[7*A_cols]);

            // shuffle doubles within 128-bit lanes
            s0 = _mm512_unpacklo_pd(r0, r1);
            s1 = _mm512_unpackhi_pd(r0, r1);
            s2 = _mm512_unpacklo_pd(r2, r3);
            s3 = _mm512_unpackhi_pd(r2, r3);
            s4 = _mm512_unpacklo_pd(r4, r5);
            s5 = _mm512_unpackhi_pd(r4, r5);
            s6 = _mm512_unpacklo_pd(r6, r7);
            s7 = _mm512_unpackhi_pd(r6, r7);

            // shuffle 2x2 blocks of doubles
            r0 = _mm512_permutex2var_pd(s0, idx_2x2_0, s2);
            r1 = _mm512_permutex2var_pd(s1, idx_2x2_0, s3);
            r2 = _mm512_permutex2var_pd(s2, idx_2x2_1, s0);
            r3 = _mm512_permutex2var_pd(s3, idx_2x2_1, s1);
            r4 = _mm512_permutex2var_pd(s4, idx_2x2_0, s6);
            r5 = _mm512_permutex2var_pd(s5, idx_2x2_0, s7);
            r6 = _mm512_permutex2var_pd(s6, idx_2x2_1, s4);
            r7 = _mm512_permutex2var_pd(s7, idx_2x2_1, s5);

            // shuffle 4x4 blocks of doubles
            s0 = _mm512_permutex2var_pd(r0, idx_4x4_0, r4);
            s1 = _mm512_permutex2var_pd(r1, idx_4x4_0, r5);
            s2 = _mm512_permutex2var_pd(r2, idx_4x4_0, r6);
            s3 = _mm512_permutex2var_pd(r3, idx_4x4_0, r7);
            s4 = _mm512_permutex2var_pd(r4, idx_4x4_1, r0);
            s5 = _mm512_permutex2var_pd(r5, idx_4x4_1, r1);
            s6 = _mm512_permutex2var_pd(r6, idx_4x4_1, r2);
            s7 = _mm512_permutex2var_pd(r7, idx_4x4_1, r3);

            // write back 8x8 block of write array
#if defined(USE_AVX_STREAMING_STORES)
            _mm512_stream_pd(&B_block[0], s0);
            _mm512_stream_pd(&B_block[A_rows], s1);
            _mm512_stream_pd(&B_block[2*A_rows], s2);
            _mm512_stream_pd(&B_block[3*A_rows], s3);
            _mm512_stream_pd(&B_block[4*A_rows], s4);
            _mm512_stream_pd(&B_block[5*A_rows], s5);
            _mm512_stream_pd(&B_block[6*A_rows], s6);
            _mm512_stream_pd(&B_block[7*A_rows], s7);
#else
            _mm512_store_pd(&B_block[0], s0);
            _mm512_store_pd(&B_block[A_rows], s1);
            _mm512_store_pd(&B_block[2*A_rows], s2);
            _mm512_store_pd(&B_block[3*A_rows], s3);
            _mm512_store_pd(&B_block[4*A_rows], s4);
            _mm512_store_pd(&B_block[5*A_rows], s5);
            _mm512_store_pd(&B_block[6*A_rows], s6);
            _mm512_store_pd(&B_block[7*A_rows], s7);
#endif
        }
    }
}
