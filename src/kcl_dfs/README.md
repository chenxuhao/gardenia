k-Clique Listing (k-CL)
================================================================================

DESCRIPTION 
--------------------------------------------------------------------------------

Author: Xuhao Chen <cxh@mit.edu>

This program counts the number of k-cliques in a given undirected graph.

This implementation reduces the search space by counting each k-clique only once.
This is done by establishing a total ordering among vertices in the input graph.
To setup this total ordering, the input undirected graph is converted into
a directed acyclic graph (DAG). This technqiue is well known as orientation.
To find k-cliques, we start from each vertex v, and iteratively add one more 
vertex from its neighborhood. We use a connectivity map to record the neighborhood
connectivity. Each time a vertex is added, the map is queried to make sure
the new vertex is connected with all vertices in the embedding.

INPUT
--------------------------------------------------------------------------------

The input graph is preprocessed internally to meet these requirements:

  - to be undirected

  - no self-loops

  - no duplicate edges (or else will be counted as multiple triangles)

  - neighborhoods are sorted by vertex identifiers

BUILD
--------------------------------------------------------------------------------

1. Run make at this directory

2. Or run make at the top-level directory

  - kcl_omp_base : one thread per vertex using OpenMP

RUN
--------------------------------------------------------------------------------

Decompress the file and put it in the 'datasets' sub-directory:

    $ cd ../../datasets
    $ tar zxvf citeseer.tar.gz

The following is an example command line:

`$ ../../bin/kcl_omp_base ../../datasets/citeseer/graph 4`

PERFORMANCE
--------------------------------------------------------------------------------

Please see details in the paper.

CITATION
--------------------------------------------------------------------------------

Please cite the following paper if you use this code:

```
@article{Pangolin,
	title={Pangolin: An Efficient and Flexible Graph Mining System on CPU and GPU},
	author={Xuhao Chen and Roshan Dathathri and Gurbinder Gill and Keshav Pingali},
	year={2020},
	journal = {Proc. VLDB Endow.},
	issue_date = {August 2020},
	volume = {13},
	number = {8},
	month = aug,
	year = {2020},
	numpages = {12},
	publisher = {VLDB Endowment},
}
```

