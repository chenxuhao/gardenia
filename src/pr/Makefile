include ../common.mk
INCLUDES+=-I $(CUB_DIR)
SRC=main.cu
OBJS=gather.o scatter.o fusion.o main.o

all: $(OBJS) pr_gather pr_scatter pr_fusion

%.o: %.cc
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $<

%.o: %.cu
	$(NVCC) $(NVFLAGS) $(INCLUDES) -c $<

pr_scatter: main.o scatter.o
	$(NVCC) $(NVFLAGS) $(INCLUDES) $(EXTRA) main.o scatter.o -o $@ $(LIBS)
	mv $@ $(BIN)

pr_gather: main.o gather.o
	$(NVCC) $(NVFLAGS) $(INCLUDES) $(EXTRA) main.o gather.o -o $@ $(LIBS)
	mv $@ $(BIN)

pr_fusion: main.o fusion.o
	$(NVCC) $(NVFLAGS) $(INCLUDES) $(EXTRA) main.o fusion.o -o $@ $(LIBS)
	mv $@ $(BIN)

clean:
	rm *.o
