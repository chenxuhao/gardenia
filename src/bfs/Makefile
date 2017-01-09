include ../common.mk
INCLUDES+=-I $(CUB_DIR)
B40C_INC=-I $(B40_DIR) -I $(B40_DIR)/test
SRC=main.cc
OBJS= topo.o base.o linear.o main.o

all: $(OBJS) bfs_base bfs_linear bfs_topo

%.o: %.cc
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $<

%.o: %.cu
	$(NVCC) $(NVFLAGS) $(INCLUDES) -c $<

bfs_base: base.o main.o
	$(NVCC) $(NVFLAGS) $(INCLUDES) $(EXTRA) main.o base.o -o $@ $(LIBS)
	mv $@ $(BIN)

bfs_linear: linear.o main.o
	$(NVCC) $(NVFLAGS) $(INCLUDES) $(EXTRA) main.o linear.o -o $@ $(LIBS)
	mv $@ $(BIN)

bfs_topo: topo.o main.o
	$(NVCC) $(NVFLAGS) $(INCLUDES) $(EXTRA) main.o topo.o -o $@ $(LIBS)
	mv $@ $(BIN)

bfs_merrill: merrill.cu main.o
	$(NVCC) $(NVFLAGS) $(INCLUDES) $(B40C_INC) -c merrill.cu -o merrill.o
	$(NVCC) $(NVFLAGS) $(INCLUDES) main.o merrill.o -o $@ $(LIBS)
	mv $@ $(BIN)

clean:
	rm *.o
