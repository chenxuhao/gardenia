include ../common.mk
INCLUDES+=-I $(CUB_DIR)
SRC=main.cu
EXE=mst

all: $(EXE)_topo

$(EXE)_wlw:
	$(NVCC) $(NVFLAGS) $(INCLUDES) $(EXTRA) $(SRC) -DVARIANT=3 -o $@ $(LIBS)

$(EXE)_wlc:
	$(NVCC) $(NVFLAGS) $(INCLUDES) $(EXTRA) $(SRC) -DVARIANT=2 -o $@ $(LIBS)

$(EXE)_topo:
	$(NVCC) $(NVFLAGS) $(INCLUDES) $(EXTRA) $(SRC) -DVARIANT=1 -o $@
	mv $@ $(BIN)

clean:
	rm $(EXE)_topo

