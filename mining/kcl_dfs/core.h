
typedef struct {
	unsigned key;
	unsigned value;
} keyvalue;

class bheap {
public:
	unsigned n_max;	// max number of nodes.
	unsigned n;	// number of nodes.
	unsigned *pt;	// pointers to nodes.
	keyvalue *kv; // nodes.
	bheap() {}
	~bheap() {
		free(pt);
		free(kv);
	}
	void construct(unsigned n_max) {
		n_max = n_max;
		n = 0;
		pt = (unsigned *)malloc(n_max*sizeof(unsigned));
		for (unsigned i = 0; i < n_max; i ++) pt[i] = -1;
		kv = (keyvalue *)malloc(n_max * sizeof(keyvalue));
	}
	void swap(unsigned i, unsigned j) {
		keyvalue kv_tmp=kv[i];
		unsigned pt_tmp=pt[kv_tmp.key];
		pt[kv[i].key]=pt[kv[j].key];
		kv[i]=kv[j];
		pt[kv[j].key]=pt_tmp;
		kv[j]=kv_tmp;
	}
	void bubble_up(unsigned i) {
		unsigned j = (i - 1) / 2;
		while (i > 0) {
			if (kv[j].value > kv[i].value) {
				swap(i ,j);
				i = j;
				j = (i - 1) / 2;
			}
			else break;
		}
	}
	void bubble_down() {
		unsigned i = 0, j1 = 1, j2 = 2, j;
		while (j1 < n) {
			j = ( (j2 < n) && (kv[j2].value < kv[j1].value) ) ? j2 : j1 ;
			if (kv[j].value < kv[i].value) {
				swap(i, j);
				i = j;
				j1 = 2*i+1;
				j2 = j1+1;
				continue;
			}
			break;
		}
	}
	void insert(keyvalue item) {
		pt[item.key] = (n) ++;
		kv[n-1] = item;
		bubble_up(n-1);
	}
	void update(unsigned key) {
		unsigned i = pt[key];
		if (i != (unsigned)-1) {
			((kv[i]).value) --;
			bubble_up(i);
		}
	}
	keyvalue popmin() {
		keyvalue min = kv[0];
		pt[min.key] = (unsigned)-1;
		kv[0] = kv[--(n)];
		pt[kv[0].key] = 0;
		bubble_down();
		return min;
	}
	//Building the heap structure with (key,value)=(node,degree) for each node
	void mkheap(unsigned n,unsigned *v) {
		construct(n);
		for (unsigned i = 0; i < n; i ++) {
			keyvalue item;
			item.key = i;
			item.value = v[i];
			insert(item);
		}
	}
};

//computing degeneracy ordering and core value
void ord_core(unsigned n, unsigned nnz, unsigned *rank, Edgelist el) {
	unsigned *d0 = (unsigned *)calloc(n,sizeof(unsigned));
	unsigned *cd0 = (unsigned *)malloc((n+1)*sizeof(unsigned));
	unsigned *adj0 = (unsigned *)malloc(2*nnz*sizeof(unsigned));
	for (unsigned i = 0; i < nnz; i ++) {
		d0[el[i].s]++;
		d0[el[i].t]++;
	}
	cd0[0] = 0;
	for (unsigned i = 1; i < n + 1; i ++) {
		cd0[i] = cd0[i-1] + d0[i-1];
		d0[i-1] = 0;
	}
	for (unsigned i = 0; i < nnz; i ++) {
		adj0[ cd0[el[i].s] + d0[ el[i].s ]++ ] = el[i].t;
		adj0[ cd0[el[i].t] + d0[ el[i].t ]++ ] = el[i].s;
	}
	bheap heap;
	heap.mkheap(n, d0);
	unsigned r = 0;
	for (unsigned i = 0; i < n; i ++) {
		keyvalue kv = heap.popmin();
		rank[kv.key] = n - (++r);
		for (unsigned j = cd0[kv.key]; j < cd0[kv.key + 1]; j ++) {
			heap.update(adj0[j]);
		}
	}
	free(d0);
	free(cd0);
	free(adj0);
}
