include ../common.mk
INCLUDES+=-I ~/cub-1.1.1
LIBS=-L$(CUDA_HOME)/lib64
#B40_DIR=./back40computing-read-only
#B40C_INC=-I$(B40_DIR) -I$(B40_DIR)/test
EXTRA=-cudart shared
SRC=main.cu

all: bfs_wlw bfs_wlc

bfs_wlw:
	$(NVCC) $(NVFLAGS) $(INCLUDES) $(EXTRA) $(SRC) -DVARIANT=1 -o $@ $(LIBS)
	mv $@ $(BIN)

bfs_wlc:
	$(NVCC) $(NVFLAGS) $(INCLUDES) $(EXTRA) $(SRC) -DVARIANT=2 -o $@ $(LIBS)
	mv $@ $(BIN)

bfs_ldb: $(B40_DIR)
	$(NVCC) $(NVFLAGS) $(INCLUDES) $(EXTRA) $(SRC) $(B40C_INC) -DVARIANT=4 -o $@ $(LIBS)

bfs_fusion:
	$(NVCC) $(NVFLAGS) $(INCLUDES) $(EXTRA) $(SRC) -DVARIANT=3 -o $@ $(LIBS)

bfs_topo:
	$(NVCC) $(NVFLAGS) $(INCLUDES) $(EXTRA) $(SRC) -DVARIANT=5 -o $@ $(LIBS)

clean:
	rm bfs_wlw bfs_wlc

