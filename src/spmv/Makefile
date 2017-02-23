include ../common.mk
CXXFLAGS+=-std=c++11
SRC=main.cc
OBJS=scalar.o verifier.o main.o

all: $(OBJS) spmv_scalar

%.o: %.cc
	$(CXX) $(CXXFLAGS) $(PARFLAG) $(INCLUDES) -c $<

%.o: %.cu
	$(NVCC) $(NVFLAGS) $(INCLUDES) -c $<

spmv_omp: omp_base.o main.o verifier.o
	$(CXX) $(CXXFLAGS) $(INCLUDES) main.o omp_base.o verifier.o -o $@ -lgomp
	mv $@ $(BIN)

spmv_scalar: main.o scalar.o verifier.o
	$(NVCC) $(NVFLAGS) $(INCLUDES) $(EXTRA) main.o scalar.o verifier.o -o $@ $(LIBS)
	mv $@ $(BIN)

spmv_lb:
	$(NVCC) $(NVFLAGS) $(INCLUDES) $(EXTRA) main.o -o $@ $(LIBS)
	mv $@ $(BIN)

clean:
	rm *.o
