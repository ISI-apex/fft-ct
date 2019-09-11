/**
 * FFT functions
 *
 * @author Connor Imes <cimes@isi.edu>
 * @date 2019-09-11
 */
#include <complex.h>
#include <errno.h>
#include <pthread.h>
#include <stdlib.h>

#include <fftw3.h>

#include "util.h"
#include "fft-threads-fftw.h"

struct fft_thread_arg {
    const fftw_plan *p;
    size_t A_rows;
    size_t r_min;
    size_t r_max;
    size_t thr_num;
};

static void ft_arg_init(struct fft_thread_arg *ft_arg, const fftw_plan *p,
                        size_t A_rows, size_t r_min, size_t r_max,
                        size_t thr_num)
{
    ft_arg->p = p;
    ft_arg->A_rows = A_rows;
    ft_arg->r_min = r_min;
    ft_arg->r_max = r_max;
    ft_arg->thr_num = thr_num;
}

static void *fft_thread_fftw(void *args)
{
    const struct fft_thread_arg *ft_arg = (const struct fft_thread_arg *)args;
    size_t i;
    for (i = ft_arg->r_min; i < ft_arg->r_max; i++)
        fftw_execute(ft_arg->p[i]);
    pthread_exit((void *)ft_arg->thr_num);
}

void fft_thr_fftw(const fftw_plan *p, size_t A_rows, size_t num_thr)
{
    size_t r_min, r_max, thr_num;
    pthread_t *threads = assert_malloc(num_thr * sizeof(pthread_t));
    struct fft_thread_arg *args = assert_malloc(num_thr * sizeof(struct fft_thread_arg));
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
        ft_arg_init(&args[thr_num], p, A_rows, r_min, r_max, thr_num);
        errno = pthread_create(&threads[thr_num], NULL, fft_thread_fftw,
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
