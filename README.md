FFT Corner Turn Benchmark
=========================

This project contains a simple FFT Corner Turn benchmark of the form:
1-D FFTs -> Matrix Transpose -> 1-D FFTs.

Prerequisites
-------------

The benchmark requires:

* [FFTW3](http://www.fftw.org/) - tested with version `3.3.8`.
* [Intel MKL](https://software.intel.com/mkl) - tested with version `2019 Update 4`.

Compiler and linker flags are discovered using `pkg-config`.

Building
--------

To build using the included Makefile:

	make

If `FFTW3` is installed to a non-standard location `${PREFIX}`, you must first
configure `PKG_CONFIG_PATH`. E.g., when `FFTW3` was configured with
`./configure --prefix=${PREFIX}`:

	export PKG_CONFIG_PATH=${PREFIX}/lib/pkgconfig/

Usage
-----

The run the benchmark:

	./fft-ct <nrows> <ncols>

E.g., to use a 2048x4096 input matrix (2048 1-D FFTs -> CT -> 4096 1-D FFTs):

	./fft-ct 2048 4096
