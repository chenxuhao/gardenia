include ../common.mk
SRC=main.cc
OBJS=vector.o omp_base.o verifier.o main.o
INCLUDES += -I${MKLROOT}/include
LIBS += -L${MKLROOT}/lib/intel64
all: $(OBJS) spmv_vector spmv_omp_base

%.o: %.cpp
	$(ICPC) $(ICPCFLAGS) $(INCLUDES) -march=core-avx2 -c $<

ifeq ($(HOST),X86)
%.o: %.cc
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $<
else
%.o: %.cc
	$(CXX) $(SIMFLAGS) $(INCLUDES) -c $<
endif

%.o: %.cu
	$(NVCC) $(NVFLAGS) $(INCLUDES) -c $<

spmv_omp_base: omp_base.o main.o verifier.o
	$(CXX) $(CXXFLAGS) $(INCLUDES) main.o omp_base.o verifier.o -o $@ -lgomp
	mv $@ $(BIN)

spmv_omp_base_sim: omp_base.o main.o verifier.o
	$(CXX) $(SIMFLAGS) $(INCLUDES) main.o omp_base.o verifier.o $(M5OP) -o $@ -lgomp
	mv $@ $(BIN)

spmv_ocl_base: ocl_base.o main.o verifier.o
	$(CXX) $(CXXFLAGS) $(INCLUDES) main.o ocl_base.o verifier.o -o $@ -lOpenCL
	mv $@ $(BIN)

spmv_omp_simd: omp_simd.o main.o verifier.o
	$(ICPC) $(ICPCFLAGS) $(INCLUDES) main.o omp_simd.o verifier.o -o $@
	mv $@ $(BIN)

spmv_base: main.o base.o verifier.o
	$(NVCC) $(NVFLAGS) $(INCLUDES) $(EXTRA) main.o base.o verifier.o -o $@ $(LIBS)
	mv $@ $(BIN)

spmv_tex: main.o tex.o verifier.o
	$(NVCC) $(NVFLAGS) $(INCLUDES) $(EXTRA) main.o tex.o verifier.o -o $@ $(LIBS)
	mv $@ $(BIN)

spmv_warp: main.o warp.o verifier.o
	$(NVCC) $(NVFLAGS) $(INCLUDES) $(EXTRA) main.o warp.o verifier.o -o $@ $(LIBS)
	mv $@ $(BIN)

spmv_vector: main.o vector.o verifier.o
	$(NVCC) $(NVFLAGS) $(INCLUDES) $(EXTRA) main.o vector.o verifier.o -o $@ $(LIBS)
	mv $@ $(BIN)

spmv_partition: main.o partition.o verifier.o
	$(NVCC) $(NVFLAGS) $(INCLUDES) $(EXTRA) main.o partition.o verifier.o -o $@ $(LIBS)
	mv $@ $(BIN)

spmv_tiling: main.o tiling.o verifier.o
	$(NVCC) $(NVFLAGS) $(INCLUDES) $(EXTRA) main.o tiling.o verifier.o -o $@ $(LIBS)
	mv $@ $(BIN)

spmv_cusparse: main.o cusparse.o verifier.o
	$(NVCC) $(NVFLAGS) $(INCLUDES) $(EXTRA) main.o cusparse.o verifier.o -o $@ $(LIBS) -lcusparse
	mv $@ $(BIN)

spmv_mkl: main.o mkl.o verifier.o
	$(ICPC) $(ICPCFLAGS) $(INCLUDES) main.o mkl.o verifier.o -o $@ $(LIBS) -lmkl_intel_lp64 -lmkl_core -lmkl_sequential -lpthread -lm
	mv $@ $(BIN)

spmv_omp_target: main.o omp_target.o verifier.o
	$(ICPC) $(ICPCFLAGS) $(INCLUDES) $(EXTRA) main.o omp_target.o verifier.o -o $@ $(LIBS)
	mv $@ $(BIN)

clean:
	rm *.o
