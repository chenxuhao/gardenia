include ../common.mk
INCLUDES+=-I $(CUB_DIR)
SRC=main.cc
OBJS=omp_base.o base.o main.o verifier.o
all: $(OBJS) tc_base tc_omp

%.o: %.cc
	$(CXX) $(CXXFLAGS) $(PARFLAG) $(INCLUDES) -c $<

%.o: %.cu
	$(NVCC) $(NVFLAGS) $(INCLUDES) -c $<

tc_base: base.o main.o verifier.o
	$(NVCC) $(NVFLAGS) $(INCLUDES) $(EXTRA) main.o base.o verifier.o -o $@ $(LIBS)
	mv $@ $(BIN)

tc_lb: lb.o main.o verifier.o
	$(NVCC) $(NVFLAGS) $(INCLUDES) $(EXTRA) main.o lb.o verifier.o -o $@ $(LIBS)
	mv $@ $(BIN)

tc_omp: omp_base.o main.o verifier.o
	$(CXX) $(CXXFLAGS) $(INCLUDES) omp_base.o main.o verifier.o -o $@ -lgomp
	mv $@ $(BIN)

clean:
	rm *.o
