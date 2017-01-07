include ../common.mk
INCLUDES+=-I $(CUB_DIR)
SRC=main.cu
EXE=mst

all: $(EXE)_topo

$(EXE)_base:
	$(NVCC) $(NVFLAGS) $(INCLUDES) $(EXTRA) $(SRC) base.o -o $@ $(LIBS)

$(EXE)_topo:
	$(NVCC) $(NVFLAGS) $(INCLUDES) $(EXTRA) $(SRC) -o $@
	mv $@ $(BIN)

clean:
	rm *.o

