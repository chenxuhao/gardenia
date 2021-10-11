#!/bin/bash
BENCHMARKS="motif_omp_automine"
BENCHMARKS="motif_omp_automine motif_omp_cmap"
GRAPHS="twitter20"
GRAPHS="citeseer"
GRAPHS="mico cit-Patents patent_citations youtube livej orkut"
OUTDIR=/home/cxh/gardenia_code/outputs
INDIR=/mnt/md0/graph_inputs
EXEDIR=/home/cxh/gardenia_code/bin
NT=32
SIZES="4"
export OMP_NUM_THREADS=$NT
#mkdir -p $OUTDIR

for BIN in $BENCHMARKS; do
  for GRAPH in $GRAPHS; do
    for K in $SIZES; do
      echo "Running $BIN with $GRAPH and size $K (nthreads=$NT) on $(date)"
      echo "Running $BIN with $GRAPH and size $K (nthreads=$NT) on $(date)" >> $OUTDIR/date.log
      echo "./bin/$BIN $INDIR/$GRAPH/graph $K &> $OUTDIR/$BIN-$GRAPH-$K-$NT.log"
      $EXEDIR/$BIN $INDIR/$GRAPH/graph $K &> $OUTDIR/$BIN-$GRAPH-$K-$NT.log
    done
  done
done
