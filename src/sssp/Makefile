include ../common.mk
INCLUDES+=-I ~/cub-1.1.1
LIBS=-L$(CUDA_HOME)/lib64
EXTRA=-cudart shared
EXE=sssp
SRC=main.cu

all: $(EXE)_wlw $(EXE)_wlc $(EXE)_topo

$(EXE)_wlw:
	$(NVCC) $(NVFLAGS) $(INCLUDES) $(EXTRA) $(SRC) -DVARIANT=1 -o $@ $(LIBS)
	mv $@ $(BIN)

$(EXE)_wlc:
	$(NVCC) $(NVFLAGS) $(INCLUDES) $(EXTRA) $(SRC) -DVARIANT=2 -o $@ $(LIBS)
	mv $@ $(BIN)

$(EXE)_topo:
	$(NVCC) $(NVFLAGS) $(INCLUDES) $(EXTRA) $(SRC) -DVARIANT=3 -o $@ $(LIBS)
	mv $@ $(BIN)

clean:
	rm $(EXE)_wlw $(EXE)_wlc $(EXE)_topo

