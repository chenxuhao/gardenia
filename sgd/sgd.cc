#include <iostream>
#include <vector>

using namespace std;
#define K 20 // dimension of the latent vector (number of features)
#define lambda 0.001
#define step 0.00000035
unsigned int rseed[16*MAX_THREADS];

void SGD(int num_users, int *row_offsets, int *column_indices, int *rating, double *lv[K], int max_iters, double epsilon = 0) {
	double *res_lv[K];
	for (int iter=0; iter < max_iters; iter++) {
		double error = 0;
#pragma omp parallel for
		for (NodeID n=0; n < g.num_nodes(); n++)
			message[n] = vertexprop;
#pragma omp parallel for
		for (NodeID u=0; u < g.num_nodes(); u++) {
			for (NodeID v : g.in_neigh(u)) {
				double estimate = 0;
				for (int i = 0; i < K; i++) {
					estimate += v.lv[i] * u.lv[i];
				}
				double error = edge_val - estimate;
				for (int i = 0; i < K; i++) res.lv[i] = v.lv[i] * error;
				for (int i = 0; i < K; i++) message_out.lv[i] += res.lv[i];
			}
		}
#pragma omp parallel for reduction(+ : error) schedule(dynamic, 64)
		for (NodeID u=0; u < g.num_nodes(); u++) {
			for (NodeID v : g.in_neigh(u))
				for (int i =0; i < K; i++) u.lv[i] += step * (-lambda * u.lv[i] + message_out.lv[i]);;
		}
		bool result = false;
#pragma omp parallel for
		for (NodeID u=0; u < g.num_nodes(); u++) {
			for (int i = 0; i < K; i++) {
				if (fabs(p.lv[i] - lv[i]) > 1e-7) {
					result = true;
				}
			}
		}
		if (result = false)
			break;
	}
}
