include ../common.mk
INCLUDES+=-I $(CUB_DIR)
B40C_INC=-I $(B40_DIR) -I $(B40_DIR)/test
SRC=main.cc
OBJS= topo.o base.o linear.o merrill.o

all: $(OBJ) bfs_base bfs_linear bfs_topo

%.o: %.cc
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $<

%.o: %.cu
	$(NVCC) $(NVFLAGS) $(INCLUDES) -c $<

bfs_base: base.o
	$(NVCC) $(NVFLAGS) $(INCLUDES) $(EXTRA) $(SRC) base.o -o $@ $(LIBS)
	mv $@ $(BIN)

bfs_linear: linear.o
	$(NVCC) $(NVFLAGS) $(INCLUDES) $(EXTRA) $(SRC) linear.o -o $@ $(LIBS)
	mv $@ $(BIN)

bfs_topo: topo.o
	$(NVCC) $(NVFLAGS) $(INCLUDES) $(EXTRA) $(SRC) topo.o -o $@ $(LIBS)
	mv $@ $(BIN)

bfs_merrill: merrill.o
	$(NVCC) $(NVFLAGS) $(INCLUDES) $(B40C_INC) $(EXTRA) $(SRC) merrill.o -o $@ $(LIBS)
	mv $@ $(BIN)

clean:
	rm *.o
