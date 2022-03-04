#ifndef SUBGRAPH_H_
#define SUBGRAPH_H_

class subgraph {
public:
	unsigned *n;//n[l]: number of vertices in G_l
	unsigned **d;//d[l]: degrees of G_l
	unsigned *adj;//truncated list of neighbors
	unsigned char *lab;//lab[i] label of vertex i
	unsigned **vertices;//sub[l]: vertices in G_l
	unsigned core;
	unsigned max_size;
	subgraph() {}
	~subgraph() {
		for (unsigned i = 2; i < max_size; i ++) {
			free(d[i]);
			free(vertices[i]);
		}
		free(n);
		free(d);
		free(vertices);
		free(lab);
		free(adj);
	}
	void allocate(unsigned c, unsigned k) {
		max_size = k;
		core = c;
		n = (unsigned *)calloc(k, sizeof(unsigned));
		d = (unsigned **)malloc(k * sizeof(unsigned*));
		vertices = (unsigned **)malloc(k * sizeof(unsigned*));
		for (unsigned i = 2; i < k; i ++) {
			d[i] = (unsigned *)malloc(core * sizeof(unsigned));
			vertices[i] = (unsigned *)malloc(core * sizeof(unsigned));
		}
		lab = (unsigned char*)calloc(core, sizeof(unsigned char));
		adj = (unsigned *)malloc(core * core * sizeof(unsigned));
	}
};
/*
void free_subgraph(subgraph *sg, unsigned char k) {
	unsigned char i;
	free(sg->n);
	for (i=2;i<k;i++) {
		free(sg->d[i]);
		free(sg->vertices[i]);
	}
	free(sg->d);
	free(sg->vertices);
	free(sg->lab);
	free(sg->adj);
	free(sg);
}

subgraph* allocsub(unsigned core, unsigned k) {
	subgraph* sg = (subgraph *)malloc(sizeof(subgraph));
	sg->n = (unsigned *)calloc(k, sizeof(unsigned));
	sg->d = (unsigned **)malloc(k * sizeof(unsigned*));
	sg->vertices = (unsigned **)malloc(k * sizeof(unsigned*));
	for (unsigned i = 2; i < k; i ++) {
		sg->d[i] = (unsigned *)malloc(core * sizeof(unsigned));
		sg->vertices[i] = (unsigned *)malloc(core * sizeof(unsigned));
	}
	sg->lab = (unsigned char*)calloc(core, sizeof(unsigned char));
	sg->adj = (unsigned *)malloc(core * core * sizeof(unsigned));
	sg->core = core;
	return sg;
}
*/

#endif
