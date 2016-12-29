// check if correctly coloured
void ColorVerifier(int m, int nnz, int *csrRowPtr, int *csrColInd, int *coloring, int *correct) {
	int i, offset, neighbor_j;
	for (i = 0; i < m; i++) {
		for (offset = csrRowPtr[i]; offset < csrRowPtr[i + 1]; offset++) {
			neighbor_j = csrColInd[offset];
			if (coloring[i] == coloring[neighbor_j] && neighbor_j != i) {
				*correct = 0;
				//printf("coloring[%d] = coloring[%d] = %d\n", i, neighbor_j, coloring[i]);
				break;
			}
		}
	}   
}

// store colour of all vertex
void write_solution(char *fname, int *coloring, int n) {
	int i;
	FILE *fp;
	fp = fopen(fname, "w");
	for (i = 0; i < n; i++) {
		//fprintf(fp, "%d:%d\n", i, coloring[i]);
		fprintf(fp, "%d\n", coloring[i]);
	}   
	fclose(fp);
}

