CC ?= gcc
CFLAGS += -Wall -std=c99
LDFLAGS +=

OBJS = transpose.o util.o
BINS = fft-ct fft-2d transp test-transpose

# fftw3
CFLAGS_FFTW = $(shell pkg-config --cflags fftw3)
LDFLAGS_FFTW = $(shell pkg-config --libs --static fftw3)
USE_FFTW ?= 1
ifeq ($(strip $(USE_FFTW)),1)
CFLAGS += $(CFLAGS_FFTW) -DUSE_FFTW
LDFLAGS += $(LDFLAGS_FFTW)
endif

# MKL (e.g., at /opt/intel/compilers_and_libraries/linux/mkl/bin/pkgconfig/)
CFLAGS_MKL = $(shell pkg-config --cflags mkl-static-ilp64-seq)
LDFLAGS_MKL = $(shell pkg-config --libs --static mkl-static-ilp64-seq)
USE_MKL ?= 1
ifeq ($(strip $(USE_MKL)),1)
CFLAGS += $(CFLAGS_MKL) -DUSE_MKL
LDFLAGS += $(LDFLAGS_MKL)
endif

.PHONY: all
all: $(BINS)

%.o: %.c Makefile $(wildcard *.h)
	$(CC) -c -o $@ $< $(CFLAGS)

fft-ct: fft-ct.o $(OBJS)
	$(CC) -o $@ $^ $(LDFLAGS)

fft-2d: fft-2d.o $(OBJS)
	$(CC) -o $@ $^ $(LDFLAGS)

transp: transp.o $(OBJS)
	$(CC) -o $@ $^ $(LDFLAGS)

test-transpose: test-transpose.o $(OBJS)
	$(CC) -o $@ $^ $(LDFLAGS)

clean:
	rm -f *.o $(BINS)
