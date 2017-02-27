include ../common.mk
INCLUDES+=-I $(CUB_DIR)
EXE=sssp
SRC=main.cc
OBJS= topo.o base.o linear.o main.o verifier.o

all: $(OBJS) $(EXE)_base $(EXE)_linear $(EXE)_topo $(EXE)_omp $(EXE)_dstep

%.o: %.cc
	$(CXX) $(CXXFLAGS) $(PARFLAG) $(INCLUDES) -c $<

%.o: %.cu
	$(NVCC) $(NVFLAGS) $(INCLUDES) -c $<

$(EXE)_omp: omp_base.o main.o verifier.o
	$(CXX) $(CXXFLAGS) $(INCLUDES) main.o omp_base.o verifier.o -o $@ -lgomp
	mv $@ $(BIN)

$(EXE)_base: main.o base.o verifier.o
	$(NVCC) $(NVFLAGS) $(INCLUDES) $(EXTRA) main.o base.o verifier.o -o $@ $(LIBS)
	mv $@ $(BIN)

$(EXE)_linear: main.o linear.o verifier.o
	$(NVCC) $(NVFLAGS) $(INCLUDES) $(EXTRA) main.o linear.o verifier.o -o $@ $(LIBS)
	mv $@ $(BIN)

$(EXE)_dstep: main.o dstep.o verifier.o
	$(NVCC) $(NVFLAGS) $(INCLUDES) $(EXTRA) main.o dstep.o verifier.o -o $@ $(LIBS)
	mv $@ $(BIN)

$(EXE)_topo: main.o topo.o verifier.o
	$(NVCC) $(NVFLAGS) $(INCLUDES) $(EXTRA) main.o topo.o verifier.o -o $@ $(LIBS)
	mv $@ $(BIN)

clean:
	rm *.o

