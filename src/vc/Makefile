include ../common.mk
INCLUDES+=-I $(CUB_DIR)
SRC=main.cc
OBJS=omp_base.o base.o topo.o bitset.o main.o verifier.o

all: $(OBJS) vc_base vc_bitset vc_topo vc_omp

%.o: %.cc
	$(CXX) $(CXXFLAGS) $(PARFLAG) $(INCLUDES) -c $<

%.o: %.cu
	$(NVCC) $(NVFLAGS) $(INCLUDES) -c $<

vc_omp: omp_base.o main.o verifier.o
	$(CXX) $(CXXFLAGS) $(INCLUDES) main.o omp_base.o verifier.o -o $@ -lgomp
	mv $@ $(BIN)

vc_topo: main.o topo.o verifier.o
	$(NVCC) $(NVFLAGS) $(INCLUDES) $(EXTRA) main.o topo.o verifier.o -o $@
	mv $@ $(BIN)

vc_base: main.o base.o verifier.o
	$(NVCC) $(NVFLAGS) $(INCLUDES) $(EXTRA) main.o base.o verifier.o -o $@
	mv $@ $(BIN)

vc_bitset: main.o bitset.o verifier.o
	$(NVCC) $(NVFLAGS) $(INCLUDES) $(EXTRA) main.o bitset.o verifier.o -o $@
	mv $@ $(BIN)

clean:
	rm *.o

