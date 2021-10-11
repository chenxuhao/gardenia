GARDENIA Benchmark Suite [![Build Status](https://travis-ci.org/chenxuhao/gardenia.svg)](https://travis-ci.org/chenxuhao/gardenia)
===================

Copyright 2020 Xuhao Chen, Massachusetts Institute of Technology

GARDENIA: Graph Analytics Repository for Designing Efficient Next-generation Accelerators

Link: https://github.com/chenxuhao/gardenia

This is the reference implementation for the [GARDENIA](https://chenxuhao.github.io/) [Benchmark Suite](https://chenxuhao.github.io/). It is designed to be a portable high-performance baseline for desgining next-generation accelerators. It uses CUDA, OpenCL and OpenMP for parallelism. The details of the benchmark can be found in the [specification](https://arxiv.org/pdf/1708.04567.pdf).

The GARDENIA Benchmark Suite is an extented version of the [GAP](https://gap.cs.berkeley.edu/) [Benchmark Suite](https://gap.cs.berkeley.edu/benchmark.html) which is intended to help graph processing research by standardizing evaluations. The benchmark provides a baseline implementation which incorporates state-of-the-art optimization techniques proposed for modern accelerators, such as GPUs and MICs. These baseline implementations are representative of state-of-the-art performance, and thus new contributions should outperform them to demonstrate an improvement. The code infrastructure is ported from GAPBS and Lonstargpu by the [ISS group](http://iss.ices.utexas.edu/?p=projects/galois/lonestargpu) at the University of Texas. 
To run the benchmarks on the GPGPU-Sim simulator, please use the [revised GPGPU-Sim version](https://github.com/chenxuhao/gpgpu-sim-ndp) and we highly recommend you to use CUDA-5.5 for the compatibility issue.
Note that in the code, 'm' is the number of vertices, and 'nnz' is the number of edges.
Graphs are stored as the CSR format in memory.
CSR is represented by two auxiliary data structures: 'row_offsets' and 'column_indices'.
You will need to download [CUB](https://nvlabs.github.io/cub/).

Graph Analytics Kernels Included
----------------

+ Betweenness Centrality (BC) - Brandes
+ Breadth-First Search (BFS) - direction optimizing
+ Connected Components (CC) - Afforest & Shiloach-Vishkin
+ Minimum Spanning Tree (MST) - 
+ PageRank (PR) - iterative method in pull direction
+ Strongly Connected Components (SCC) - Forward-Backward-Trim
+ Stochastic Gradient Descent (SGD) -
+ Sparse Matrix-Vector Multiplication (SpMV)
+ Single-Source Shortest Paths (SSSP) - delta stepping
+ Symmetric Gauss-seidel Smoother (SymGS) -
+ Vertex Coloring (VC) - Gebremedhin and Manne

Graph Mining Kernels Included
----------------

+ Triangle Counting (TC) - orientation and merge-based intersection
+ k-Clique Listing (k-CL) - orientation and connectivity map
+ Subgraph Listing (SL) - matching order and partial orders
+ k-Motif Counting (k-MC) - matching order and partial orders
+ Frequent Subgraph Mining (FSM) - early pruning with anti-monotonicity

Quick Start
-----------

Setup CUB library:

    $ git submodule update --init --recursive

Setup environment variables:

    $ cd src
    $ cp common.mk.example common.mk
    $ vim common.mk // modify this file to setup the compilation

Build the project (you will need to install gcc and nvcc first):

    $ make

Or go to each sub-directory, e.g. src/bfs, and then

    $ make

Download datasets from the UFSMC or SNAP website:

    $ wget https://www.cise.ufl.edu/research/sparse/MM/SNAP/soc-LiveJournal1.tar.gz

Decompress the dataset file and put it in the 'datasets' sub-directory:

    $ tar zxvf soc-LiveJournal1.tar.gz
    $ mv soc-LiveJournal1.mtx datasets/

Find out commandline format by running executable without argument:

    $ cd bin
    $ ./bfs_linear_base
    Usage: ./bfs_linear_base <filetype> <graph-prefix> [symmetrize(0/1)] [reverse(0/1)] [source_id(0)]

Run BFS on a directed graph starting from vertex 0:

    $ cd bin
    $ ./bfs_linear_base mtx ../datasets/soc-LiveJournal1 0 0 0

To run on CPU or Intel Xeon Phi coprocessor, set the following environment variable:

    $ export OMP_NUM_THREADS=[ number of cores in system ]


Graph Loading
-------------

The graph loading infrastructure understands the following formats:

+ `.mtx` [Matrix Market](https://math.nist.gov/MatrixMarket/formats.html) format

+ `.gr` [9th DIMACS Implementation Challenge](https//www.dis.uniroma1.it/challenge9/download.shtml) format

+ `.graph` Metis format (used in [10th DIMACS Implementation Challenge](https://www.cc.gatech.edu/dimacs10/index.shtml))


How to Cite
-----------

Author: 
[Xuhao Chen](https://chenxuhao.github.io) <cxh@mit.edu>

Please cite this code by the benchmark specification:

Zhen Xu, Xuhao Chen, Jie Shen, Yang Zhang, Cheng Chen, Canqun Yang,
[GARDENIA: A Domain-specific Benchmark Suite for Next-generation Accelerators](https://arxiv.org/pdf/1708.04567.pdf), 
ACM Journal on Emerging Technologies in Computing Systems, 2018.

Xuhao Chen, Roshan Dathathri, Gurbinder Gill, Keshav Pingali,
[Pangolin: An Efficient and Flexible Graph Mining System on CPU and GPU](https://arxiv.org/pdf/1911.06969.pdf),
PVLDB 13(8): 1190-1205, 2020

Other citations:

Xuhao Chen, Cheng Chen, Jie Shen, Jianbin Fang, Tao Tang, Canqun Yang, Zhiying Wang,
[Orchestrating Parallel Detection of Strongly Connected Components on GPUs](https://chenxuhao.github.io/docs/parco-scc.pdf), 
Parallel Computing, Vol 78, Pages 101–114, 2018.

Xuhao Chen, Pingfan Li, Jianbin Fang, Tao Tang, Zhiying Wang, Canqun Yang,
[Efficient and High-quality Sparse Graph Coloring on the GPU](https://arxiv.org/pdf/1606.06025v1.pdf), 
Concurrency and Computation: Practice and Experience, Volume 29, Issue 10, 17 April 2017.

Pingfan Li, Xuhao Chen et al., 
[High Performance Detection of Strongly Connected Components in Sparse Graphs on GPUs](https://chenxuhao.github.io/docs/pmam-2017.pdf), 
In the Proceedings of the International Workshop on Programming Models and Applications for Multicores and Manycores, in conjunction with the 22nd ACM SIGPLAN Symposium on Principles and Practice of Parallel Programming (PPoPP), Austin, TX, Feb 2017 

Pingfan Li, Xuhao Chen, Zhe Quan, Jianbin Fang, Huayou Su, Tao Tang, Canqun Yang,
[High Performance Parallel Graph Coloring on GPGPUs](https://chenxuhao.github.io/docs/ipdpsw-2016.pdf), 
In the Proceedings of the 30th IEEE International Parallel & Distributed Processing Symposium Workshop (IPDPSW), Chicago, IL, May 2016

More documentation coming soon. For questions, please email <cxh.nudt@gmail.com>
