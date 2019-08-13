/**
 * Utility functions
 *
 * @author Connor Imes <cimes@isi.edu>
 * @date 2019-07-15 
 */
#ifndef UTIL_H
#define UTIL_H

#include <stdlib.h>

float rand_flt(void);
double rand_dbl(void);

void fill_rand_flt(float *a, size_t len);
void fill_rand_dbl(double *a, size_t len);

void matrix_print_flt(const float *A, size_t nrows, size_t ncols);
void matrix_print_dbl(const double *A, size_t nrows, size_t ncols);

int is_eq_flt(float a, float b);
int is_eq_dbl(double a, double b);

void *assert_malloc(size_t sz);
void *assert_malloc_al(size_t sz);

#endif /* UTIL_H */
