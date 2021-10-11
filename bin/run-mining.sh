INDIR=/mnt/md0/graph_inputs
INDIR=/h2/xchen/work/AMruntime/graph_converter
NT=56
K=4
export OMP_NUM_THREADS=$NT
BIN=tc_omp_base
GRAPH=orkut
echo "Running $BIN with $GRAPH (nthreads=$NT) on $(date)"
./bin/$BIN $INDIR/$GRAPH/graph

BIN=kcl_omp_base
GRAPH=livej
printf "\n"
echo "Running $BIN with $GRAPH (k=$K,nthreads=$NT) on $(date)"
./bin/$BIN $INDIR/$GRAPH/graph $K

BIN=sgl_omp_base
GRAPH=youtube
PT=diamond
printf "\n"
echo "Running $BIN with $GRAPH (pattern=$PT,nthreads=$NT) on $(date)"
./bin/$BIN $INDIR/$GRAPH/graph $PT

BIN=motif_omp_formula
GRAPH=mico
printf "\n"
echo "Running $BIN with $GRAPH (k=$K,nthreads=$NT) on $(date)"
./bin/$BIN $INDIR/$GRAPH/graph $K

BIN=motif_omp_base
GRAPH=patent_citations
printf "\n"
echo "Running $BIN with $GRAPH (k=$K,nthreads=$NT) on $(date)"
./bin/$BIN $INDIR/$GRAPH/graph $K

