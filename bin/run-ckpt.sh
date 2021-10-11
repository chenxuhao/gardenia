#!/bin/bash

BIN=kcl_omp_sim
BIN=motif_omp_base_sim
BIN=sgl_omp_sim
GRAPH=patent_citations
GRAPH=youtube
GRAPH=mico
PT=4
PT=rectangle
NT=1
TICK=3000000 # mico no DAG
TICK=3300000 # patent no DAG
TICK=5000000 # youtube no DAG
OUTDIR=/home/cxh/gem5/outputs
CKPTDIR=/home/cxh/gem5/checkpoints/$BIN-$GRAPH-$PT-$NT/

mkdir -p $CKPTDIR

echo "Running $BIN with $GRAPH (pattern=$PT,nthreads=$NT) on $(date) ($OUTDIR/$BIN-$GRAPH-$PT-$NT.log)"
build/X86/gem5.fast \
  configs/example/se.py \
  --cmd="/home/cxh/gardenia_code/bin/$BIN" \
  -o "/graph_inputs/$GRAPH/graph $PT" \
  --checkpoint-dir=$CKPTDIR \
  --num-cpus=$NT --mem-size=4GB \
  &> $OUTDIR/$BIN-$GRAPH-$PT-$NT.log
#  --take-checkpoints=$TICK --at-instruction 
echo "Done $BIN with $GRAPH (pattern=$PT,nthreads=$NT) on $(date)"
