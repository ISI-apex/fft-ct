CC ?= gcc
CFLAGS += -Wall -std=c99
LDFLAGS +=

OBJS = transpose.o util.o
BINS = test-transpose

# fftw3
LIB_FFTW = fftw3
USE_FFTW ?= $(shell pkg-config --exists $(LIB_FFTW) && echo 1)
ifeq ($(strip $(USE_FFTW)),1)
CFLAGS_FFTW = $(shell pkg-config --cflags $(LIB_FFTW))
LDFLAGS_FFTW = $(shell pkg-config --libs --static $(LIB_FFTW))
CFLAGS += $(CFLAGS_FFTW) -DUSE_FFTW
LDFLAGS += $(LDFLAGS_FFTW)
OBJS += transpose-fftw.o util-fftw.o
BINS += fft-ct fft-2d transp
endif

# MKL (e.g., at /opt/intel/compilers_and_libraries/linux/mkl/bin/pkgconfig/)
LIB_MKL = mkl-static-ilp64-seq
USE_MKL ?= $(shell pkg-config --exists $(LIB_MKL) && echo 1)
ifeq ($(strip $(USE_MKL)),1)
CFLAGS_MKL = $(shell pkg-config --cflags $(LIB_MKL))
LDFLAGS_MKL = $(shell pkg-config --libs --static $(LIB_MKL))
CFLAGS += $(CFLAGS_MKL) -DUSE_MKL
LDFLAGS += $(LDFLAGS_MKL)
OBJS += transpose-mkl.o util-mkl.o
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
