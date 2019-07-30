CC ?= gcc
CFLAGS += -Wall -std=c99
LDFLAGS +=

OBJS = transpose.o util.o
BINS = fft-ct fft-2d transp test-transpose

# fftw3
CFLAGS += $(shell pkg-config --cflags fftw3)
LDFLAGS += $(shell pkg-config --libs --static fftw3)

# MKL (e.g., at /opt/intel/compilers_and_libraries/linux/mkl/bin/pkgconfig/)
CFLAGS += $(shell pkg-config --cflags mkl-static-ilp64-seq)
LDFLAGS += $(shell pkg-config --libs --static mkl-static-ilp64-seq)

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
