NVCC=nvcc
EXE=sssp
NVFLAGS=-g

all:
	$(NVCC) $(NVFLAGS) $(EXE).cu -o $(EXE) -lnvgraph

clean:
	rm $(EXE)
