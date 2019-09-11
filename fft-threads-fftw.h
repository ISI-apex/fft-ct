/**
 * FFT functions
 *
 * @author Connor Imes <cimes@isi.edu>
 * @date 2019-09-11
 */
#ifndef FFT_THREADS_FFTW_H
#define FFT_THREADS_FFTW_H

#include <complex.h>
#include <stdlib.h>

#include <fftw3.h>

void fft_thr_fftw(const fftw_plan *p, size_t A_rows, size_t num_thr);

#endif /* FFT_THREADS_FFTW_H */
