#!/bin/bash
BENCHMARKS="sgl_omp_automine sgl_omp_cmap"
GRAPHS="citeseer"
GRAPHS="mico cit-Patents patent_citations youtube livej orkut"
#GRAPHS="twitter20 friendster"
OUTDIR=/home/cxh/gardenia_code/outputs
INDIR=/mnt/md0/graph_inputs
EXEDIR=/home/cxh/gardenia_code/bin
NT=32
PATTERNS="diamond rectangle"
export OMP_NUM_THREADS=$NT
#mkdir -p $OUTDIR

for BIN in $BENCHMARKS; do
  for PT in $PATTERNS; do
    for GRAPH in $GRAPHS; do
      echo "Running $BIN with $GRAPH and size $PT (nthreads=$NT) on $(date)"
      echo "Running $BIN with $GRAPH and size $PT (nthreads=$NT) on $(date)" >> $OUTDIR/date.log
      echo "./bin/$BIN $INDIR/$GRAPH/graph $PT &> $OUTDIR/$BIN-$GRAPH-$PT-$NT.log"
      $EXEDIR/$BIN $INDIR/$GRAPH/graph $PT &> $OUTDIR/$BIN-$GRAPH-$PT-$NT.log
    done
  done
done
