FFT Corner Turn Benchmarks
==========================

This project contains benchmarks related to FFT Corner Turns.


Prerequisites
-------------

For full functionality, this project requires the following.

A C compiler that supports AVX-512 instructions, e.g., one of:

* [GCC](https://gcc.gnu.org/) >= `5.0.0` - tested with version `9.1.0`.
* [Clang](https://clang.llvm.org/) >= `3.9.1` - tested with version `10.0.1`.
* [Intel C Compiler](https://software.intel.com/c-compilers) >= `15.0.1` -
tested with version `19.0`.

Build tools:

* [CMake](https://cmake.org/) >= `2.8.12`
* [pkg-config](https://www.freedesktop.org/wiki/Software/pkg-config/)

Libraries:

* POSIX Threads - pthreads-compatible library (usually included with compiler).
* [FFTW3](http://www.fftw.org/) - tested with version `3.3.8`.
Both single and double precision libraries are used (`libfftw3f`, `libfftw3`).
* [Intel MKL](https://software.intel.com/mkl) - tested with version
`2019 Update 4`.


Building
--------

The project uses CMake:

	mkdir build
	cd build
	cmake ..
	make

If dependencies are installed to a non-standard location `${PREFIX}`, you must
first configure `PKG_CONFIG_PATH`.
E.g., if `FFTW3` is configured with `./configure --prefix=${PREFIX}`:

	export PKG_CONFIG_PATH=${PREFIX}/lib/pkgconfig/

To use a different C compiler than your environment's `cc`, configure CMake's
`CMAKE_C_COMPILER` variable, e.g.:

	cmake .. -DCMAKE_C_COMPILER=/path/to/cc


Benchmarks
----------

There are a handful benchmark program templates:

* `transp`: Populate a matrix and perform a transpose.
* `fft-2d`: Populate a matrix and perform a 2-D FFT.
Whether a transpose is actually performed depends on the FFT implementation.
* `fft-ct`: Populate a matrix and perform 1-D FFTs -> transpose -> 1-D FFTs.
In this benchmark, a transpose is always performed.


Data Types
----------

C99 types:

* `float` (flt)
* `double` (dbl)
* `float complex` (fcmplx)
* `double complex` (dcmplx)

FFTW types:

* `fftwf_complex` (fftwf) - redefined as `float complex`
* `fftw_complex` (fftw) - redefined as `double complex`

MKL types:

* `MKL_Complex8` (cmplx8) - redefined as `float complex`
* `MKL_Complex16` (cmplx16) - redefined as `double complex`


Usage
-----

Benchmark templates are used to generate benchmarks supporting a variety of data
types and transpose implementations using different algorithms and library APIs.
Benchmark names are generally in the form:
`${prog}-${datatype}-${algo}[-${lib}]`.

All benchmarks support the `-h` parameter to print a usage/help message.

Benchmarks require specifying, at a minimum, the matrix row and column count.
E.g., to perform a naive transpose of a 2048x4096 matrix with `double` data:

	./transp-dbl-naive -r 2048 -c 4096

Some implementations have constraints on parameters:

* Blocked transposes must use block dimensions that are divisors of their
corresponding matrix dimensions.  I.e., partial blocks are not supported.
* Transposes using AVX-512 instructions require matrix sizes to be multiples of
8x8 blocks.
This constraint extends to threaded AVX-512 implementations -- each thread's
partition of a matrix must be a multiple of 8x8, e.g., while a single thread (or
even three threads) may transpose a 24x24 matrix, two threads cannot because
data is partitioned evenly between threads (12x24 or 24x12, for two threads).
