#ifndef OMP_TARGET_CONFIG_H_
#define OMP_TARGET_CONFIG_H_

static void warm_up() {
	int i, n = 8;
	int *x, *y, *z;
	x = (int *) malloc(n * sizeof(int));
	y = (int *) malloc(n * sizeof(int));
	z = (int *) malloc(n * sizeof(int));
	for (i = 0; i != n; i++) { x[i] = 1; y[i] = 1; z[i] = 0; }
	#pragma omp target map(from:z[0:n]) map(to:y[0:n],x[0:n])
	#pragma omp parallel for
	for (i = 0; i < 8; ++i) z[i] = x[i] + y[i];
}

#endif
