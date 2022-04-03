#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#define N 1024

void main() {
	int n = N;
	int a = 2;
	int i, begin;
	int *x, *y, *z;
	x = (int *) malloc(n * sizeof(int));
	y = (int *) malloc(n * sizeof(int));
	z = (int *) malloc(n * sizeof(int));
	for (i = 0; i != n; i++) {
		x[i] = i;
		y[i] = 2 * i;
		z[i] = 0.0;
	}
	//for (i = 0; i < n; i++)
	//    x[i] = i;
#pragma omp target map(from:z[0:n]) map(to:y[0:n],x[0:n])
#pragma omp parallel for
	for (i = 0; i < N; ++i)
		z[i] = a * x[i] + y[i];

	printf("Checking Result...\n");
	for (i = 0; i < n; ++i)
		if (((a * x[i] + y[i])) != z[i]) {
			printf("Result is wrong at %d!!!\n", i);
			break;
		}
	printf("Result checked!!!\n");
	return;
}

