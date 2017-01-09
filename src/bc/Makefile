include ../common.mk
SRC=main.cc
OBJS=omp_base.o base.o main.o
all: $(OBJS) bc_base bc_omp

%.o: %.cc
	$(CXX) $(CXXFLAGS) $(PARFLAG) $(INCLUDES) -c $<

%.o: %.cu
	$(NVCC) $(NVFLAGS) $(INCLUDES) -c $<

bc_omp: omp_base.o main.o
	$(CXX) $(CXXFLAGS) $(INCLUDES) main.o omp_base.o -o $@ -lgomp
	mv $@ $(BIN)

bc_base: base.o main.o
	$(NVCC) $(NVFLAGS) $(INCLUDES) $(EXTRA) main.o base.o -o $@ $(LIBS)
	mv $@  $(BIN)

bc_lb:
	$(NVCC) $(NVFLAGS) $(INCLUDES) $(EXTRA) $(SRC) -o $@ $(LIBS)
	mv $@ $(BIN)

clean:
	rm *.o
