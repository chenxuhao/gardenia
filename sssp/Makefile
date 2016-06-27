include ../common.mk
INCLUDES+=-I ~/cub-1.1.1
LIBS=-L$(CUDA_HOME)/lib64
#B40_DIR=./back40computing-read-only
#B40C_INC=-I$(B40_DIR) -I$(B40_DIR)/test
EXTRA=-cudart shared
EXE=sssp
SRC=main.cu

all: $(EXE)_wlw $(EXE)_wlc

$(EXE)_wlw:
	$(NVCC) $(NVFLAGS) $(INCLUDES) $(EXTRA) $(SRC) -DVARIANT=1 -o $@ $(LIBS)
	mv $@ $(BIN)

$(EXE)_wlc:
	$(NVCC) $(NVFLAGS) $(INCLUDES) $(EXTRA) $(SRC) -DVARIANT=2 -o $@ $(LIBS)
	mv $@ $(BIN)

$(EXE)_ldb: $(B40_DIR)
	$(NVCC) $(NVFLAGS) $(INCLUDES) $(EXTRA) $(SRC) $(B40C_INC) -DVARIANT=4 -o $@ $(LIBS)

$(EXE)_fusion:
	$(NVCC) $(NVFLAGS) $(INCLUDES) $(EXTRA) $(SRC) -DVARIANT=3 -o $@ $(LIBS)

clean:
	rm $(EXE)_wlw $(EXE)_wlc

