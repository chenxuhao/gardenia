include ../common.mk
INCLUDES+=-I $(CUB_DIR)
SRC=main.cu
OBJS=gather.o gather_warp.o scatter.o fusion.o main.o verifier.o omp_base.o

all: $(OBJS) pr_gather pr_gather_warp pr_omp_base

%.o: %.cpp
	$(ICPC) $(ICPCFLAGS) $(INCLUDES) -c $<

%.o: %.cc
	#$(CXX) $(CXXFLAGS) $(INCLUDES) -c $<
	$(CXX) $(SIMFLAGS) $(INCLUDES) -c $<

%.o: %.cu
	$(NVCC) $(NVFLAGS) $(INCLUDES) -c $<

pr_omp_base: omp_base.o main.o verifier.o pr.h
	$(CXX) $(CXXFLAGS) $(INCLUDES) main.o omp_base.o verifier.o -o $@ -lgomp
	mv $@ $(BIN)

pr_omp_base_sim: omp_base.o main.o verifier.o
	$(CXX) $(SIMFLAGS) $(INCLUDES) main.o omp_base.o verifier.o $(M5OP) -o $@ -lgomp
	mv $@ $(BIN)

pr_scatter: main.o scatter.o verifier.o pr.h
	$(NVCC) $(NVFLAGS) $(INCLUDES) $(EXTRA) main.o scatter.o verifier.o -o $@ $(LIBS)
	mv $@ $(BIN)

pr_gather: main.o gather.o verifier.o pr.h
	$(NVCC) $(NVFLAGS) $(INCLUDES) $(EXTRA) main.o gather.o verifier.o -o $@ $(LIBS)
	mv $@ $(BIN)

pr_gather_warp: main.o gather_warp.o verifier.o pr.h
	$(NVCC) $(NVFLAGS) $(INCLUDES) $(EXTRA) main.o gather_warp.o verifier.o -o $@ $(LIBS)
	mv $@ $(BIN)

pr_fusion: main.o fusion.o verifier.o pr.h
	$(NVCC) $(NVFLAGS) $(INCLUDES) $(EXTRA) main.o fusion.o verifier.o -o $@ $(LIBS)
	mv $@ $(BIN)

pr_omp_target: main.o omp_target.o verifier.o
	$(ICPC) $(CXXFLAGS) $(INCLUDES) $(EXTRA) main.o omp_target.o verifier.o -o $@ $(LIBS)
	mv $@ $(BIN)

clean:
	rm *.o
