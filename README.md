FFT Corner Turn Benchmark
=========================

This project contains a simple FFT Corner Turn benchmark of the form:
1-D FFTs -> Transpose -> 1D FFTs.

Prerequisites
-------------

The benchmark requires the [FFTW3](http://www.fftw.org/) library and development
files, which are discovered using `pkg-config`.

Building
--------

To build using the included Makefile:

	make

If `FFTW3` is installed to a non-standard location `${PREFIX}`, you must first
configure `PKG_CONFIG_PATH`. E.g., when `FFTW3` was configured with
`./configure --prefix=${PREFIX}`:

	export PKG_CONFIG_PATH=${PREFIX}/lib/pkgconfig/
