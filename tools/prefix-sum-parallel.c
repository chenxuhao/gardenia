#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/resource.h>
#include <sys/time.h>
#include <omp.h>
#include <time.h>
#include <vector>
#include <cassert>

void display(std::vector<int> t) {
	for (int i = 0; i < t.size(); ++i)
		printf("%d ", t[i]);
	printf("\n");
}

void prefix_sum(std::vector<int> &T) {
	size_t n = T.size();
	assert(n > 1);
	int i;
	if (n==2) T[1] += T[0];
	else {
#pragma omp parallel num_threads(n) private(i)
		{
			int tid = omp_get_thread_num();
			for(i=2; i <= n; i*=2) {
				if ((tid % i) == (i - 1))
					T[tid] += T[tid - i/2];
#pragma omp barrier
			}
			for(i=i/2; i >1; i/=2) {
				if ((tid % i) == (i/2 - 1))
					if (!(tid < (i/2 - 1)))
						T[tid] += T[tid - i/2];
#pragma omp barrier
			}
		}
	}
}

std::vector<int> PrefixSum(std::vector<int> input, int num_threads) {
	int n = input.size();
	int level = 2;
	while(level<=n) {
		int untill = n/level;
		omp_set_num_threads(num_threads);
		#pragma omp parallel for
		for(int i=0; i<untill; i++) {
			int work_for = level*(i+1)-1;
			int get = work_for - (level/2);
			input[work_for] += input[get];
		}
		level *=2;
	}
	int save = input[n-1];
	input[n-1] = 0;
	while(level!=1) {
		int untill = n/level;
		omp_set_num_threads(num_threads);
		#pragma omp parallel for
		for(int i=0;i<untill;i++) {
			int work_for = level*(i+1)-1;
			int get = work_for - (level/2);
			int temp = input[work_for];
			input[work_for] += input[get];
			input[get] =  temp;
		}
		level /= 2;
	}
	input.push_back(save);
	input.erase(input.begin());
	return input;
}

int main(int argc, char const *argv[]) {
	if (argc < 2) {
		printf("USAGE : %s <N>\n", argv[0]);
		exit(1);
	}
	int n = atoi(argv[1]);
	int nthreads = atoi(argv[2]);
	std::vector<int> tab(n, 0);
	double start;
	double end;
	srand (time(NULL));
	for (int i = 0; i < n; ++i) tab[i] = rand()%10;
	display(tab);
	printf("start SP2 with n = %d\n", n);
	start = omp_get_wtime();
	std::vector<int> output = PrefixSum(tab, nthreads);
	end = omp_get_wtime();
	display(output);
	printf("user time used for SP2 : %f\n", end - start);
	return 0;
}

