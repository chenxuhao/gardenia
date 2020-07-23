#include <vector>

#ifdef GTX1080_PB
#define BITS 13 // 2^13 = 8K elements = 32K bytes
#else
#ifdef GTX1050_PB
#define BITS 12 // 2^12 = 4K elements = 16K bytes
#else
//#define BITS 16 // 64K elements = 256K bytes
#define BITS 12 // 2^12 = 4K elements = 16K bytes
//#define BITS 13 // 2^13 = 8K elements = 32K bytes
#endif
#endif

#define BIN_WIDTH (1 << BITS)

#ifdef ALIGNED
vector<aligned_vector<IndexT> > vertex_bins(num_bins);
vector<aligned_vector<ScoreT> > value_bins(num_bins);
#else
vector<vector<IndexT> > vertex_bins;
vector<vector<ScoreT> > value_bins;
#endif
//vector<int> dest_bids;
vector<int> sizes;
vector<int> pos;
vector<ScoreT *> addr;

void preprocessing(int m, int nnz, IndexT *row_offsets, IndexT *column_indices) {
	printf("Start preprocessing ...\n");
	int num_bins = (m-1) / BIN_WIDTH + 1;
	printf("bin width: %d, number of bins: %d\n", BIN_WIDTH, num_bins);
	vertex_bins.resize(num_bins);
	value_bins.resize(num_bins);
	sizes.resize(num_bins);
	//dest_bids.resize(nnz);
	pos.resize(nnz);
	addr.resize(nnz);
	std::fill(sizes.begin(), sizes.end(), 0);
	//std::fill(dest_bids.begin(), dest_bids.end(), 0);
	std::fill(pos.begin(), pos.end(), 0);
	//std::fill(addr.begin(), addr.end(), NULL);

	Timer t;
	t.Start();
	for (int u = 0; u < m; u ++) {
		const IndexT row_begin = row_offsets[u];
		const IndexT row_end = row_offsets[u+1];
		for (IndexT offset = row_begin; offset < row_end; offset ++) {
			IndexT v = column_indices[offset];
			int dest_bin = v >> BITS; // v / BIN_WIDTH
			int bin_size = vertex_bins[dest_bin].size();
			vertex_bins[dest_bin].resize(bin_size+1);
			value_bins[dest_bin].resize(bin_size+1);
			vertex_bins[dest_bin][sizes[dest_bin]] = v;
			//dest_bids[offset] = dest_bin;
			pos[offset] = sizes[dest_bin];
			addr[offset] = value_bins[dest_bin].data() + sizes[dest_bin];
			sizes[dest_bin] ++;
		}
	}
	t.Stop();
	printf("\truntime [preprocessing] = %f ms.\n", t.Millisecs());
}

