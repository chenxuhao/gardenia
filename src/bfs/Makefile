include ../common.mk
INCLUDES+=-I $(CUB_DIR)
ICPCFLAGS+=-std=c++98
B40C_INC=-I $(B40_DIR) -I $(B40_DIR)/test
SRC=main.cc
#OBJS= topo_base.o linear_base.o topo_vector.o linear_vector.o topo_lb.o linear_lb.o bottom_up.o hybrid_base.o hybrid_lb.o main.o verifier.o omp_base.o omp_beamer.o omp_topo.o
OBJS= linear_lb.o main.o verifier.o omp_base.o omp_beamer.o

#all: $(OBJS) bfs_linear_base bfs_linear_lb bfs_topo_base bfs_topo_lb bfs_omp_beamer bfs_omp_base bfs_hybrid_lb
all: $(OBJS) bfs_linear_lb bfs_omp_beamer bfs_omp_base

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

bfs_omp_base: omp_base.o main.o verifier.o
	$(CXX) $(CXXFLAGS) $(INCLUDES) main.o omp_base.o verifier.o -o $@ -lgomp
	mv $@ $(BIN)

bfs_omp_base_sim: omp_base.o main.o verifier.o
	$(CXX) $(SIMFLAGS) $(INCLUDES) main.o omp_base.o verifier.o $(M5OP) -o $@ -lgomp
	mv $@ $(BIN)

bfs_omp_topo: omp_topo.o main.o verifier.o
	$(CXX) $(CXXFLAGS) $(INCLUDES) main.o omp_topo.o verifier.o -o $@ -lgomp
	mv $@ $(BIN)

bfs_omp_beamer: omp_beamer.o main.o verifier.o
	$(CXX) $(CXXFLAGS) $(INCLUDES) main.o omp_beamer.o verifier.o -o $@ -lgomp
	mv $@ $(BIN)

bfs_omp_target: main.o omp_target.o verifier.o
	$(ICPC) $(CXXFLAGS) $(INCLUDES) $(EXTRA) main.o omp_target.o verifier.o -o $@ $(LIBS)
	mv $@ $(BIN)

bfs_linear_base: linear_base.o main.o verifier.o
	$(NVCC) $(NVFLAGS) $(INCLUDES) $(EXTRA) main.o linear_base.o verifier.o -o $@ $(LIBS)
	mv $@ $(BIN)

bfs_linear_vector: linear_vector.o main.o verifier.o
	$(NVCC) $(NVFLAGS) $(INCLUDES) $(EXTRA) main.o linear_vector.o verifier.o -o $@ $(LIBS)
	mv $@ $(BIN)

bfs_linear_lb: linear_lb.o main.o verifier.o
	$(NVCC) $(NVFLAGS) $(INCLUDES) $(EXTRA) main.o linear_lb.o verifier.o -o $@ $(LIBS)
	mv $@ $(BIN)

bfs_topo_base: topo_base.o main.o verifier.o
	$(NVCC) $(NVFLAGS) $(INCLUDES) $(EXTRA) main.o topo_base.o verifier.o -o $@ $(LIBS)
	mv $@ $(BIN)

bfs_topo_vector: topo_vector.o main.o verifier.o
	$(NVCC) $(NVFLAGS) $(INCLUDES) $(EXTRA) main.o topo_vector.o verifier.o -o $@ $(LIBS)
	mv $@ $(BIN)

bfs_topo_lb: topo_lb.o main.o verifier.o
	$(NVCC) $(NVFLAGS) $(INCLUDES) $(EXTRA) main.o topo_lb.o verifier.o -o $@ $(LIBS)
	mv $@ $(BIN)

bfs_fusion: fusion.o main.o verifier.o
	$(NVCC) $(NVFLAGS) $(INCLUDES) $(EXTRA) main.o fusion.o verifier.o -o $@ $(LIBS)
	mv $@ $(BIN)

bfs_hybrid_base: hybrid_base.o main.o verifier.o
	$(NVCC) $(NVFLAGS) $(INCLUDES) $(EXTRA) main.o hybrid_base.o verifier.o -o $@ $(LIBS)
	mv $@ $(BIN)

bfs_hybrid_vector: hybrid_vector.o main.o verifier.o
	$(NVCC) $(NVFLAGS) $(INCLUDES) $(EXTRA) main.o hybrid_vector.o verifier.o -o $@ $(LIBS)
	mv $@ $(BIN)

bfs_hybrid_lb: hybrid_lb.o main.o verifier.o
	$(NVCC) $(NVFLAGS) $(INCLUDES) $(EXTRA) main.o hybrid_lb.o verifier.o -o $@ $(LIBS)
	mv $@ $(BIN)

bfs_afree: atomic_free.o main.o verifier.o
	$(NVCC) $(NVFLAGS) $(INCLUDES) $(EXTRA) main.o atomic_free.o verifier.o -o $@ $(LIBS)
	mv $@ $(BIN)

bfs_bu: bottom_up.o main.o verifier.o
	$(NVCC) $(NVFLAGS) $(INCLUDES) $(EXTRA) main.o bottom_up.o verifier.o -o $@ $(LIBS)
	mv $@ $(BIN)

bfs_merrill: merrill.cu main.o verifier.o
	$(NVCC) $(NVFLAGS) $(INCLUDES) $(B40C_INC) -c merrill.cu -o merrill.o
	$(NVCC) $(NVFLAGS) $(INCLUDES) main.o merrill.o verifier.o -o $@ $(LIBS)
	mv $@ $(BIN)

clean:
	rm *.o
