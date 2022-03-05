include ../common.mk
INCLUDES+=-I $(CUB_DIR)
ICPCFLAGS+=-std=c++98
B40_DIR=../../back40computing-read-only
B40C_INC=-I $(B40_DIR) -I $(B40_DIR)/test
OBJS= main.o verifier.o 
all: bfs_linear_base bfs_linear_lb bfs_omp_beamer bfs_omp_base

%.o: %.cpp
	$(ICPC) $(ICPCFLAGS) $(INCLUDES) -c $<

ifeq ($(HOST),X86)
%.o: %.cc
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $<
else
%.o: %.cc
	$(CXX) $(SIMFLAGS) $(INCLUDES) -c $<
endif

%.o: %.cu
	$(NVCC) $(NVFLAGS) $(INCLUDES) -c $<

bfs_ocl_base: ocl_base.o $(OBJS)
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(OBJS) ocl_base.o -o $@ -lOpenCL
	mv $@ $(BIN)

bfs_omp_base: omp_base.o $(OBJS)
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(EXTRA) $(OBJS) omp_base.o -o $@ -lgomp
	mv $@ $(BIN)

bfs_omp_topo: omp_topo.o $(OBJS)
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(OBJS) omp_topo.o -o $@ -lgomp
	mv $@ $(BIN)

bfs_omp_beamer: omp_beamer.o $(OBJS)
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(OBJS) omp_beamer.o -o $@ -lgomp
	mv $@ $(BIN)

bfs_omp_target: omp_target.o $(OBJS)
	$(ICPC) $(CXXFLAGS) $(INCLUDES) $(EXTRA) $(OBJS) omp_target.o -o $@ $(LIBS)
	mv $@ $(BIN)

bfs_linear_base: linear_base.o $(OBJS)
	$(NVCC) $(NVFLAGS) $(INCLUDES) $(EXTRA) $(OBJS) linear_base.o -o $@ $(LIBS)
	mv $@ $(BIN)

bfs_linear_pb: linear_pb.o $(OBJS)
	$(NVCC) $(NVFLAGS) $(INCLUDES) $(EXTRA) $(OBJS) linear_pb.o -o $@ $(LIBS)
	mv $@ $(BIN)

bfs_linear_vector: linear_vector.o $(OBJS)
	$(NVCC) $(NVFLAGS) $(INCLUDES) $(EXTRA) $(OBJS) linear_vector.o -o $@ $(LIBS)
	mv $@ $(BIN)

bfs_linear_lb: linear_lb.o $(OBJS)
	$(NVCC) $(NVFLAGS) $(INCLUDES) $(EXTRA) $(OBJS) linear_lb.o -o $@ $(LIBS)
	mv $@ $(BIN)

bfs_topo_base: topo_base.o $(OBJS)
	$(NVCC) $(NVFLAGS) $(INCLUDES) $(EXTRA) $(OBJS) topo_base.o -o $@ $(LIBS)
	mv $@ $(BIN)

bfs_topo_vector: topo_vector.o $(OBJS)
	$(NVCC) $(NVFLAGS) $(INCLUDES) $(EXTRA) $(OBJS) topo_vector.o -o $@ $(LIBS)
	mv $@ $(BIN)

bfs_topo_lb: topo_lb.o $(OBJS)
	$(NVCC) $(NVFLAGS) $(INCLUDES) $(EXTRA) $(OBJS) topo_lb.o -o $@ $(LIBS)
	mv $@ $(BIN)

bfs_topo_pb: topo_pb.o $(OBJS)
	$(NVCC) $(NVFLAGS) $(INCLUDES) $(EXTRA) $(OBJS) topo_pb.o -o $@ $(LIBS)
	mv $@ $(BIN)

bfs_fusion: fusion.o $(OBJS)
	$(NVCC) $(NVFLAGS) $(INCLUDES) $(EXTRA) $(OBJS) fusion.o -o $@ $(LIBS)
	mv $@ $(BIN)

bfs_hybrid_base: hybrid_base.o $(OBJS)
	$(NVCC) $(NVFLAGS) $(INCLUDES) $(EXTRA) $(OBJS) hybrid_base.o -o $@ $(LIBS)
	mv $@ $(BIN)

bfs_hybrid_vector: hybrid_vector.o $(OBJS)
	$(NVCC) $(NVFLAGS) $(INCLUDES) $(EXTRA) $(OBJS) hybrid_vector.o -o $@ $(LIBS)
	mv $@ $(BIN)

bfs_hybrid_lb: hybrid_lb.o $(OBJS)
	$(NVCC) $(NVFLAGS) $(INCLUDES) $(EXTRA) $(OBJS) hybrid_lb.o -o $@ $(LIBS)
	mv $@ $(BIN)

bfs_hybrid_tile: hybrid_tile.o $(OBJS)
	$(NVCC) $(NVFLAGS) $(INCLUDES) $(EXTRA) $(OBJS) hybrid_tile.o -o $@ $(LIBS)
	mv $@ $(BIN)

bfs_afree: atomic_free.o $(OBJS)
	$(NVCC) $(NVFLAGS) $(INCLUDES) $(EXTRA) $(OBJS) atomic_free.o -o $@ $(LIBS)
	mv $@ $(BIN)

bfs_bu: bottom_up.o $(OBJS)
	$(NVCC) $(NVFLAGS) $(INCLUDES) $(EXTRA) $(OBJS) bottom_up.o -o $@ $(LIBS)
	mv $@ $(BIN)

bfs_merrill: merrill.cu main.o verifier.o
	$(NVCC) $(NVFLAGS) $(INCLUDES) $(B40C_INC) -c merrill.cu -o merrill.o
	$(NVCC) $(NVFLAGS) $(INCLUDES) main.o merrill.o verifier.o -o $@ $(LIBS)
	mv $@ $(BIN)

clean:
	rm *.o
