/**
 * Transpose functions.
 *
 * @author Kaushik Datta <kdatta@isi.edu>
 * @date 2019-08-06
 */
void transpose_flt_threads_row(const float* restrict A, float* restrict B,
			       size_t A_rows, size_t A_cols,
			       size_t n_thr);

void transpose_dbl_threads_row(const double* restrict A, double* restrict B,
			       size_t A_rows, size_t A_cols,
			       size_t num_thr);

void transpose_flt_threads_col(const float* restrict A, float* restrict B,
			       size_t A_rows, size_t A_cols,
			       size_t num_thr);

void transpose_dbl_threads_col(const double* restrict A, double* restrict B,
			       size_t A_rows, size_t A_cols,
			       size_t num_thr);
