#include "common.h"
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <string.h>

static const int FIELD_LENGTH = 128;
static const float MAX_RANDOM_VAL = 10.0f;

struct Coordinate {
	int x; 
	int y; 
	float val; 
};

inline int coordcmp(const void *v1, const void *v2)
{
	struct Coordinate *c1 = (struct Coordinate *) v1;
	struct Coordinate *c2 = (struct Coordinate *) v2; 

	if (c1->x != c2->x) 
	{
		return (c1->x - c2->x); 
	}
	else 
	{
		return (c1->y - c2->y); 
	}
}

// ****************************************************************************
// Function: readMatrix
//
// Purpose:
//   Reads a sparse matrix from a file of Matrix Market format 
//   Returns the data structures for the CSR format
//
// Arguments:
//   filename: c string with the name of the file to be opened
//   weight_ptr: input - pointer to uninitialized pointer
//            output - pointer to array holding the non-zero values
//                     for the  matrix 
//   column_indices_ptr: input - pointer to uninitialized pointer
//             output - pointer to array of column indices for each
//                      element of the sparse matrix
//   row_offsets_ptr: input - pointer to uninitialized pointer
//                  output - pointer to array holding
//                           indices to rows of the matrix
//   nnz: input - pointer to uninitialized int
//      output - pointer to an int holding the number of non-zero
//               elements in the matrix
//   m: input - pointer to uninitialized int
//         output - pointer to an int holding the number of rows in
//                  the matrix 
//
// Programmer: Lukasz Wesolowski
// Creation: July 2, 2010
// Returns:  nothing directly
//           allocates and returns *weight_ptr, *column_indices_ptr, and
//           *row_offsets_ptr indirectly 
//           returns m and nnz indirectly through pointers
// ****************************************************************************

void readMatrix(char *filename, int *m, int *nnz, int **row_offsets_ptr, int **column_indices_ptr, ValueType **weight_ptr) {
	std::string line;
	char id[FIELD_LENGTH];
	char object[FIELD_LENGTH]; 
	char format[FIELD_LENGTH]; 
	char field[FIELD_LENGTH]; 
	char symmetry[FIELD_LENGTH]; 

	std::ifstream mfs( filename );
	if( !mfs.good() )
	{
		std::cerr << "Error: unable to open matrix file " << filename << std::endl;
		exit( 1 );
	}

	int symmetric = 0; 
	int pattern = 0; 

	int nRows, nCols, nElements;  

	struct Coordinate *coords;

	// read matrix header
	if( getline( mfs, line ).eof() )
	{
		std::cerr << "Error: file " << filename << " does not store a matrix" << std::endl;
		exit( 1 );
	}

	sscanf(line.c_str(), "%s %s %s %s %s", id, object, format, field, symmetry); 

	if (strcmp(object, "matrix") != 0) 
	{
		fprintf(stderr, "Error: file %s does not store a matrix\n", filename); 
		exit(1); 
	}

	if (strcmp(format, "coordinate") != 0)
	{
		fprintf(stderr, "Error: matrix representation is dense\n"); 
		exit(1); 
	} 

	if (strcmp(field, "pattern") == 0) 
	{
		pattern = 1; 
		printf("This is a pattern graph!\n");
	}

	if (strcmp(symmetry, "symmetric") == 0) 
	{
		symmetric = 1; 
		printf("This is a symmetric graph!\n");
	}

	while (!getline( mfs, line ).eof() )
	{
		if (line[0] != '%') 
		{
			break; 
		}
	} 

	// read the matrix size and number of non-zero elements
	sscanf(line.c_str(), "%d %d %d", &nRows, &nCols, &nElements); 
	printf("m=%d, n=%d, nnz=%d\n", nRows, nCols, nElements);

	int valSize = nElements * sizeof(struct Coordinate);
	if (symmetric) 
	{
		valSize*=2; 
	}                     
	coords = new Coordinate[valSize]; 

	int index = 0; 
	while (!getline( mfs, line ).eof() )
	{
		if (pattern) 
		{
			sscanf(line.c_str(), "%d %d", &coords[index].x, &coords[index].y); 
			// assign a random value 
			//coords[index].val = ((ValueType) MAX_RANDOM_VAL * (rand() / (RAND_MAX + 1.0)));
			coords[index].val = 1;
		}
		else 
		{
			// read the value from file
			sscanf(line.c_str(), "%d %d %f", &coords[index].x, &coords[index].y, &coords[index].val); 
		}

		// convert into index-0-as-start representation
		coords[index].x--;
		coords[index].y--;    
		index++; 

		// add the mirror element if not on main diagonal
		if (symmetric && coords[index-1].x != coords[index-1].y) 
		{
			coords[index].x = coords[index-1].y; 
			coords[index].y = coords[index-1].x; 
			coords[index].val = coords[index-1].val; 
			index++;
		}
	}  

	nElements = index; 
	// sort the elements
	qsort(coords, nElements, sizeof(struct Coordinate), coordcmp); 

	// create CSR data structures
	*nnz = nElements; 
	*m = nRows; 
	*weight_ptr = new ValueType[nElements]; 
	*column_indices_ptr = new int[nElements];
	*row_offsets_ptr = new int[nRows+1]; 

	ValueType *weight = *weight_ptr; 
	int *column_indices = *column_indices_ptr; 
	int *row_offsets = *row_offsets_ptr; 

	row_offsets[0] = 0; 
	row_offsets[nRows] = nElements; 
	int r=0; 
	for (int i=0; i<nElements; i++) 
	{
		while (coords[i].x != r) 
		{
			row_offsets[++r] = i; 
		}
		weight[i] = coords[i].val; 
		column_indices[i] = coords[i].y;    
	}

	r = 0; 

	delete[] coords;
}

