include ../common.mk
INCLUDES+=-I ~/cub-1.1.1
LIBS=-L$(CUDA_HOME)/lib64
B40_DIR=~/back40computing-read-only
B40C_INC=-I $(B40_DIR) -I $(B40_DIR)/test
EXTRA=-cudart shared
SRC=main.cu

all: bfs_wlw bfs_wlc bfs_topo bfs_merrill

bfs_wlw:
	$(NVCC) $(NVFLAGS) $(INCLUDES) $(EXTRA) $(SRC) -DVARIANT=1 -o $@ $(LIBS)
	mv $@ $(BIN)

bfs_wlc:
	$(NVCC) $(NVFLAGS) $(INCLUDES) $(EXTRA) $(SRC) -DVARIANT=2 -o $@ $(LIBS)
	mv $@ $(BIN)

bfs_topo:
	$(NVCC) $(NVFLAGS) $(INCLUDES) $(EXTRA) $(SRC) -DVARIANT=3 -o $@ $(LIBS)
	mv $@ $(BIN)

bfs_merrill:
	$(NVCC) $(NVFLAGS) $(INCLUDES) $(B40C_INC) $(B40C_DIR) $(EXTRA) $(SRC) -DVARIANT=4 -o $@ $(LIBS)
	mv $@ $(BIN)

clean:
	rm bfs_wlw bfs_wlc bfs_topo
