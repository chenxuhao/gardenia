#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include <sys/time.h>
#include <unistd.h>
#include <math.h>
#include <float.h>
#include <omp.h>

#ifdef DP
#define REAL double
#else
#define REAL float
#endif

//int NB= BSIZE;
int BSIZE;
#define HACKFOR 128
#define max(a,b)( ((a) > (b)) ? (a) : (b) )

double cclock_( void ) {
	const  double  micro = 1.0e-06;    /* Conversion constant */
	static long    start = 0L, startu;
	struct timeval tp;                 /* Structure used by gettimeofday */
	double         wall_time;          /* To hold the result */
	if ( gettimeofday( &tp, NULL) == -1 )
		wall_time = -1.0e0;
	else if( !start ) {
		start  = tp.tv_sec;
		startu = tp.tv_usec;
		wall_time = 0.0e0;
	}
	else
		wall_time = (double) (tp.tv_sec - start) + micro*(tp.tv_usec - startu);
	return wall_time;
}

double cclock( void ) {
	return cclock_();
}

void prtspeed( int m, int l, int n, int nb, double time, int ok, unsigned long nops ) {
	double speed = 1.0e-9*nops/time;
	printf("%dx%d,%dx%d,%.4lf,%.4lf\n",m,n,nb,nb,speed,time);
	printf("Matrix size: %dx%d\n", m, n);
	printf("Block size: %dx%d\n", nb, nb);
#ifdef DP
	printf("Precision type: Double\n");
#else
	printf("Precision type: Simple\n");
#endif
	printf("  GFLOPS : %.4lf\n", speed);
	printf("  computation time (in seconds): %.4lf\n", time);
	if ( ok == 0 ) {
		printf("  Verification: Ok\n");
	} else {
		printf("  Verification: Failed  (%d)\n", ok);
	}
}

int check( int nrep, int m, int l, int n, int mDIM, int nDIM, REAL **c/*[][nDIM*BSIZE] */) {
	double eps, tvalue = (double)l;
	int    i, j, k, o, ok = 0;
	eps = 2.0*l*l*DBL_EPSILON;
	int perfectM = m / BSIZE;
	int perfectN = n / BSIZE;
	int leftOutM = m % BSIZE;
	int leftOutN = n % BSIZE;
	for(i=0;i<mDIM;i++){
		for(j=0;j<nDIM;j++){
			for(k=0;k<BSIZE;k++){
				for(o=0;o<BSIZE;o++){
					if( i == mDIM-1 && mDIM > perfectM && k >= leftOutM )
						break;
					else if( j == nDIM-1 && nDIM > perfectN && o >= leftOutN )
						break;
					else {
						if ( fabs( tvalue - (c[i*nDIM+j][k*BSIZE+o]/nrep) ) > eps ) {
							ok++;
							//printf("Bad result at [%d][%d] : expected %f but found %f\n", i*nDIM+j, k*BSIZE+o, tvalue, c[i*nDIM+j][k*BSIZE+o]);
						}
					}
				}
			}
		}
	}
	return( ok );
}

void gendat(int mDIM, int lDIM, int nDIM, int m, int l, int n, REAL **tileA, REAL **tileB, REAL **tileC) {
	int i,j,k,y;
	REAL currentValue;
	int perfectM = m / BSIZE;
	int perfectL = l / BSIZE;
	int perfectN = n / BSIZE;
	int leftOutM = m % BSIZE;
	int leftOutL = l % BSIZE;
	int leftOutN = n % BSIZE;
	for( i = 0; i < mDIM; ++i )
		for( j = 0; j < lDIM; ++j )
			for( k = 0; k < BSIZE; ++k  )
			{
				currentValue = j*BSIZE;
				for( y = 0; y < BSIZE; ++y )
				{
					if( i == mDIM-1 && mDIM > perfectM && k >= leftOutM )
						tileA[ i*lDIM + j ][ k*BSIZE+y ] = 0.0;
					else if( j == lDIM-1 && lDIM > perfectL && y >= leftOutL )
						tileA[ i*lDIM + j ][ k*BSIZE+y ] = 0.0;
					else
						tileA[ i*lDIM + j ][ k*BSIZE+y ] = ++currentValue;
				}
			}
	for( i = 0; i < lDIM; ++i )
		for( j = 0; j < nDIM; ++j )
		{
			currentValue = (i*BSIZE) + 1;
			for( k = 0; k < BSIZE; ++k,  currentValue += 1)
				for( y = 0; y < BSIZE; ++y )
				{
					if( i == lDIM-1 && lDIM > perfectL && k >= leftOutL )
						tileB[ i*nDIM + j ][ k*BSIZE+y ] = 0.0;
					else if( j == nDIM-1 && nDIM > perfectN && y >= leftOutN )
						tileB[ i*nDIM + j ][ k*BSIZE+y ] = 0.0;
					else
						tileB[ i*nDIM + j ][ k*BSIZE+y ] = 1.0 / currentValue;
				}
		}
	for( i = 0; i < lDIM; ++i ) {
		for( j = 0; j < nDIM; ++j ) {
			tileC[i][j] = 0.0;
		}
	}
}

void matmul_tile(REAL *__restrict A, REAL *__restrict B, REAL *__restrict C, int NB) {
	int nb = BSIZE;
	int i, j, k;
	//int asd[4];
	//int numa_nodes;
	int chunk = NB / omp_get_num_threads();

#pragma omp target map(tofrom: C[0:NB*NB]) map(to: A[0:NB*NB], B[0:NB*NB])
#pragma omp parallel for private(i,j,k)
	for (i = 0; i < NB; i++) {
		int g = chunk;
#pragma omp parallel for
		for (j = 0; j < NB; j++) {
			REAL sum = C[i * NB + j];
#pragma omp simd reduction(+:sum)
			for (k = 0; k < NB; k++) {
				sum += (A[i * NB + k] * B[k * NB + j]);
			}
			C[i * NB + j] = sum;
		}
	}
}


void matmul(int m, int l, int n, int mDIM, int lDIM, int nDIM, REAL **tileA, REAL **tileB, REAL **tileC) {
	int i, j, k;
	printf("==> %d %d %d \n", nDIM, mDIM, lDIM);
	for (i = 0; i < mDIM; i++) {
		for (j = 0; j < nDIM; j++) {
			for (k = 0; k < lDIM; k++) {
				matmul_tile(tileA[i * lDIM + k], tileB[k * nDIM + j], tileC[i * nDIM + j], BSIZE);
			}
		}
	}
}

int calcdim(int x) {
	int dimval;
	if(x%BSIZE != 0)
		dimval = x/BSIZE + 1;
	else
		dimval = x/BSIZE;
	return dimval;
}

int main(int argc, char* argv[]) { 
	int      lda, m, l, n;
	int      mDIM, lDIM, nDIM;
	int      ok, nrep;
	unsigned long nops;
	int      i,k,j,o;
	REAL   **a, **b, **c;
	double   time;
	FILE     *inl;
	if (2 > argc)
		BSIZE = HACKFOR;
	else
		BSIZE = atoi(argv[1]);
	inl = fopen( "test.in", "r" );
	if (inl == 0) {
		printf("No input file 'test.in' found.\n");
		exit(1);
	}
	while( ( fscanf( inl, "%d%d%d%d\n", &m, &l, &n, &nrep ) != EOF ) ){
		lda = l + 1;
		mDIM = calcdim(m);
		lDIM = calcdim(l);
		nDIM = calcdim(n);
		posix_memalign((REAL **)&a, getpagesize(), mDIM * lDIM * sizeof( REAL *));
		posix_memalign((REAL **)&b, getpagesize(), lDIM * nDIM * sizeof( REAL *));
		posix_memalign((REAL **)&c, getpagesize(), mDIM * nDIM * sizeof( REAL *));
		for (i = 0; i < mDIM * lDIM; i++)
			posix_memalign((REAL **)&a[i],getpagesize(),BSIZE * BSIZE * sizeof(REAL) );
		for (i = 0; i < lDIM * nDIM; i++)
			posix_memalign((REAL **)&b[i],getpagesize(),BSIZE * BSIZE * sizeof(REAL) );
		for (i = 0; i < mDIM * nDIM; i++)
			posix_memalign((REAL **)&c[i],getpagesize(),BSIZE * BSIZE * sizeof(REAL) );
#pragma omp register([mDIM*lDIM]a)
#pragma omp register([lDIM*nDIM]b)
#pragma omp register([mDIM*nDIM]c)
		gendat( mDIM, lDIM, nDIM, m, l, n, a, b, c );
		printf("multiply start\n");
		time = cclock();
		for( i = 0; i < nrep; i++ ){
			matmul( m, l, n, mDIM, lDIM, nDIM, a, b, c ); 
			//			#pragma omp taskwait// noflush
		}
#pragma omp taskwait
		time = cclock() - time;
		printf("multiply done\n");
		ok   = check( nrep, m, l, n, mDIM, nDIM, c);
		time = time/nrep;
		nops = (unsigned long) 2*m*l*n;
		prtspeed( m, l, n, BSIZE, time, ok, nops );
		for(i=0;i<mDIM*lDIM;i++)
			free( a[i] );
		for(i=0;i<lDIM*nDIM;i++)
			free( b[i] );
		for(i=0;i<mDIM*nDIM;i++)
			free( c[i] );
		free( a ); free( b ); free( c );
	}
	return 0;
}
