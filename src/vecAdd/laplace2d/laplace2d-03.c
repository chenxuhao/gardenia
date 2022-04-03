#include <math.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#define NN 4096
#define NM 4096
//__attribute__ ((target(mic))) 
double A[NN][NM];
//__attribute__ ((target(mic))) 
double Anew[NN][NM];
int main(int argc, char** argv) {
	//int nThreads = atoi(argv[1]);
	//omp_set_num_threads(nThreads);
	const int n = NN;
    const int m = NM;
    const int iter_max = 200;
    const double tol = 1.0e-6;
    double error = 1.0;
    memset(A, 0, n * m * sizeof(double));
    memset(Anew, 0, n * m * sizeof(double));
    for (int j = 0; j < n; j++) {
        A[j][0]    = 1.0;
        Anew[j][0] = 1.0;
    }
    printf("Jacobi relaxation Calculation: %d x %d mesh\n", n, m);
    double st = omp_get_wtime();
    int iter = 0;
#pragma omp target data map(alloc:Anew) map(A)
    while ( error > tol && iter < iter_max ) {
        error = 0.0;
#pragma omp target teams distribute parallel for reduction(max:error)
        for( int j = 1; j < n-1; j++) {
            for( int i = 1; i < m-1; i++ ) {
                Anew[j][i] = 0.25 * (A[j][i+1] + A[j][i-1] + A[j-1][i] + A[j+1][i]);
                error = fmax(error, fabs(Anew[j][i] - A[j][i]));
            }
        }
#pragma omp target teams distribute parallel for
        for( int j = 1; j < n-1; j++) {
            for( int i = 1; i < m-1; i++ ) {
                A[j][i] = Anew[j][i];    
            }
        }
        if(iter % 100 == 0) printf("%5d, %0.6f\n", iter, error);
        iter++;
    }
    double et = omp_get_wtime();
    printf(" total: %f s\n", (et - st));
    return 0;
}
