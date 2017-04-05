#ifndef BITSET_H
#define BITSET_H
void fwd_reach(int m, int *out_row_offsets, int *out_column_indices, unsigned *colors, unsigned char *status, int *scc_root);
void fwd_reach_lb(int m, int *out_row_offsets, int *out_column_indices, unsigned char *status, int *scc_root);
void bwd_reach(int m, int *in_row_offsets, int *in_column_indices, unsigned *colors, unsigned char *status);
void bwd_reach_lb(int m, int *in_row_offsets, int *in_column_indices, unsigned char *status);
void iterative_trim(int m, int *in_row_offsets, int *in_column_indices, int *out_row_offsets, int *out_column_indices, unsigned *colors, unsigned char *status, int *scc_root);
void first_trim(int m, int *in_row_offsets, int *in_column_indices, int *out_row_offsets, int *out_column_indices, unsigned char *status);
void trim2(int m, int *in_row_offsets, int *in_column_indices, int *out_row_offsets, int *out_column_indices, unsigned *colors, unsigned char *status, int *scc_root);
bool update(int m, unsigned *colors, unsigned char *status, unsigned *locks, int *scc_root);
void update_colors(int m, unsigned *colors, unsigned char *status);
void find_removed_vertices(int m, unsigned char *status, int *mark);
void print_statistics(int m, int *scc_root, unsigned char *status);

__host__ __device__ inline bool is_bwd_visited(unsigned char states) {
	return (states & 1);
}

__host__ __device__ inline bool is_fwd_visited(unsigned char states) {
	return (states & 2);
}

__host__ __device__ inline bool is_removed(unsigned char states) {
	return (states & 4);
}

__host__ __device__ inline bool is_trimmed(unsigned char states) {
	return (states & 8);
}

__host__ __device__ inline bool is_pivot(unsigned char states) {
	return (states & 16);
}

__host__ __device__ inline bool is_bwd_extended(unsigned char states) {
	return (states & 32);
}

__host__ __device__ inline bool is_fwd_extended(unsigned char states) {
	return (states & 64);
}

__host__ __device__ inline bool is_bwd_front(unsigned char states) {
	return ((states & 37) == 1);
}

__host__ __device__ inline bool is_fwd_front(unsigned char states) {
	return ((states & 70) == 2);
}

__host__ __device__ inline void set_bwd_visited(unsigned char *states) {
	*states |= 1;
}

__host__ __device__ inline void set_fwd_visited(unsigned char *states) {
	*states |= 2;
}

__host__ __device__ inline void set_removed(unsigned char *states) {
	*states |= 4;
}

__host__ __device__ inline void set_trimmed(unsigned char *states) {
	*states |= 8;
}

__host__ __device__ inline void set_pivot(unsigned char *states) {
	*states |= 16;
}

__host__ __device__ inline void set_bwd_expanded(unsigned char *states) {
	*states |= 32;
}

__host__ __device__ inline void set_fwd_expanded(unsigned char *states) {
	*states |= 64;
}

#endif
