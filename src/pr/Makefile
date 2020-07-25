include ../common.mk
INCLUDES+=-I$(CUB_DIR)
CXXFLAGS+=-std=c++11 -march=core-avx2
SRC=main.cu
OBJS=base.o warp.o main.o verifier.o omp_base.o

all: $(OBJS) pr_base pr_warp pr_omp_base

%.o: %.cpp
	$(ICPC) $(ICPCFLAGS) $(INCLUDES) -c $<

ifeq ($(HOST),X86)
%.o: %.cc pr.h
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $<
else
%.o: %.cc pr.h
	$(CXX) $(SIMFLAGS) $(INCLUDES) -c $<
endif

%.o: %.cu pr.h
	$(NVCC) $(NVFLAGS) $(INCLUDES) -c $<

pr_ocl_base: ocl_base.o main.o verifier.o
	$(CXX) $(CXXFLAGS) $(INCLUDES) main.o ocl_base.o verifier.o -o $@ -lOpenCL
	mv $@ $(BIN)

pr_omp_base: omp_base.o main.o verifier.o
	$(CXX) $(CXXFLAGS) $(INCLUDES) main.o omp_base.o verifier.o -o $@ -lgomp
	mv $@ $(BIN)

pr_omp_push: omp_push.o main.o verifier.o
	$(CXX) $(CXXFLAGS) $(INCLUDES) main.o omp_push.o verifier.o -o $@ -lgomp
	mv $@ $(BIN)

pr_pb: pb.o main.o verifier.o
	$(CXX) $(CXXFLAGS) $(INCLUDES) main.o pb.o verifier.o -o $@ -lgomp
	mv $@ $(BIN)

pr_omp_pb: omp_pb.o main.o verifier.o
	$(CXX) $(CXXFLAGS) $(INCLUDES) main.o omp_pb.o verifier.o -o $@ -lgomp
	mv $@ $(BIN)

pr_omp_delta: omp_delta.o main.o verifier.o
	$(CXX) $(CXXFLAGS) $(INCLUDES) main.o omp_delta.o verifier.o -o $@ -lgomp
	mv $@ $(BIN)

pr_omp_partition: omp_partition.o main.o verifier.o
	$(CXX) $(CXXFLAGS) $(INCLUDES) main.o omp_partition.o verifier.o -o $@ -lgomp
	mv $@ $(BIN)

pr_omp_sim: omp_sim.o main.o verifier.o
	$(CXX) $(SIMFLAGS) $(INCLUDES) main.o omp_sim.o verifier.o $(M5OP) -o $@ -lgomp
	mv $@ $(BIN)

pr_omp_target: main.o omp_target.o verifier.o
	$(ICPC) $(CXXFLAGS) $(INCLUDES) $(EXTRA) main.o omp_target.o verifier.o -o $@ $(LIBS)
	mv $@ $(BIN)

pr_push: main.o push.o verifier.o
	$(NVCC) $(NVFLAGS) $(INCLUDES) $(EXTRA) main.o push.o verifier.o -o $@ $(LIBS)
	mv $@ $(BIN)

pr_push_pb: main.o push_pb.o verifier.o
	$(NVCC) $(NVFLAGS) $(INCLUDES) $(EXTRA) main.o push_pb.o verifier.o -o $@ $(LIBS)
	mv $@ $(BIN)

pr_base: main.o base.o verifier.o
	$(NVCC) $(NVFLAGS) $(INCLUDES) $(EXTRA) main.o base.o verifier.o -o $@ $(LIBS)
	mv $@ $(BIN)

pr_warp: main.o warp.o verifier.o
	$(NVCC) $(NVFLAGS) $(INCLUDES) $(EXTRA) main.o warp.o verifier.o -o $@ $(LIBS)
	mv $@ $(BIN)

pr_vector: main.o vector.o verifier.o
	$(NVCC) $(NVFLAGS) $(INCLUDES) $(EXTRA) main.o vector.o verifier.o -o $@ $(LIBS)
	mv $@ $(BIN)

pr_lb: main.o lb.o verifier.o
	$(NVCC) $(NVFLAGS) $(INCLUDES) $(EXTRA) main.o lb.o verifier.o -o $@ $(LIBS)
	mv $@ $(BIN)

pr_delta: main.o delta.o verifier.o
	$(NVCC) $(NVFLAGS) $(INCLUDES) $(EXTRA) main.o delta.o verifier.o -o $@ $(LIBS)
	mv $@ $(BIN)

pr_fusion: main.o fusion.o verifier.o
	$(NVCC) $(NVFLAGS) $(INCLUDES) $(EXTRA) main.o fusion.o verifier.o -o $@ $(LIBS)
	mv $@ $(BIN)

pr_partition: main.o partition.o verifier.o
	$(NVCC) $(NVFLAGS) $(INCLUDES) $(EXTRA) main.o partition.o verifier.o -o $@ $(LIBS)
	mv $@ $(BIN)

pr_tiling: tiling.o main.o verifier.o
	$(NVCC) $(NVFLAGS) $(INCLUDES) $(EXTRA) main.o tiling.o verifier.o -o $@ $(LIBS)
	mv $@ $(BIN)

pr_push_tile: main.o push_tile.o verifier.o
	$(NVCC) $(NVFLAGS) $(INCLUDES) $(EXTRA) main.o push_tile.o verifier.o -o $@ $(LIBS)
	mv $@ $(BIN)

pr_nvgraph: main.o nvgraph.o verifier.o
	$(NVCC) $(NVFLAGS) $(INCLUDES) $(EXTRA) main.o nvgraph.o verifier.o -o $@ $(LIBS) -lnvgraph
	mv $@ $(BIN)

clean:
	rm *.o
