include ../common.mk
INCLUDES+=-I $(CUB_DIR)
SRC=main.cu
OBJS=gather.o scatter.o fusion.o main.o verifier.o

all: $(OBJS) pr_gather pr_scatter pr_fusion pr_omp

%.o: %.cc
	$(CXX) $(CXXFLAGS) $(PARFLAG) $(INCLUDES) -c $<

%.o: %.cu
	$(NVCC) $(NVFLAGS) $(INCLUDES) -c $<

pr_omp: omp_base.o main.o verifier.o
	$(CXX) $(CXXFLAGS) $(INCLUDES) main.o omp_base.o verifier.o -o $@ -lgomp
	mv $@ $(BIN)

pr_scatter: main.o scatter.o verifier.o
	$(NVCC) $(NVFLAGS) $(INCLUDES) $(EXTRA) main.o scatter.o verifier.o -o $@ $(LIBS)
	mv $@ $(BIN)

pr_gather: main.o gather.o verifier.o
	$(NVCC) $(NVFLAGS) $(INCLUDES) $(EXTRA) main.o gather.o verifier.o -o $@ $(LIBS)
	mv $@ $(BIN)

pr_fusion: main.o fusion.o verifier.o
	$(NVCC) $(NVFLAGS) $(INCLUDES) $(EXTRA) main.o fusion.o verifier.o -o $@ $(LIBS)
	mv $@ $(BIN)

clean:
	rm *.o
