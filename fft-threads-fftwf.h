/**
 * FFT functions
 *
 * @author Connor Imes <cimes@isi.edu>
 * @date 2019-09-11
 */
#ifndef FFT_THREADS_FFTWF_H
#define FFT_THREADS_FFTWF_H

#include <complex.h>
#include <stdlib.h>

#include <fftw3.h>

void fft_thr_fftwf(const fftwf_plan *p, size_t A_rows, size_t num_thr);

#endif /* FFT_THREADS_FFTWF_H */
