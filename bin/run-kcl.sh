#!/bin/bash
BENCHMARKS="kcl_omp_automine_dag kcl_omp_automine_nodag"
BENCHMARKS="kcl_omp_cmap_dag kcl_omp_automine_dag"
GRAPHS="citeseer"
GRAPHS="livej"
GRAPHS="twitter20"
GRAPHS="mico cit-Patents patent_citations youtube livej orkut"
OUTDIR=/home/cxh/gardenia_code/outputs
INDIR=/mnt/md0/graph_inputs
EXEDIR=/home/cxh/gardenia_code/bin
NT=32
SIZES="5"
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
