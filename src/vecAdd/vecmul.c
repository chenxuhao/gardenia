#include "stdio.h"
#define N 1024

int main() {
	int i;
	float p[N], v1[N], v2[N];
	for(i=0; i<N; i++) {
		v1[i] = 2.0;
		v2[i] = 3.0;
	}
#pragma omp target map(to:v1, v2) map(from:p)
#pragma omp parallel for
	for(i=0; i<N; i++)
	{
		p[i] = v1[i] * v2[i];
	}
	printf("output: p[0]=%f\n", p[0]);
	printf("output: p[1]=%f\n", p[1]);
	return 0;
}

