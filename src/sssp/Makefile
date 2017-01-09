include ../common.mk
INCLUDES+=-I $(CUB_DIR)
EXE=sssp
SRC=main.cc
OBJS= topo.o base.o linear.o main.o

all: $(OBJS) $(EXE)_base $(EXE)_linear $(EXE)_topo

%.o: %.cc
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $<

%.o: %.cu
	$(NVCC) $(NVFLAGS) $(INCLUDES) -c $<

$(EXE)_base: main.o base.o
	$(NVCC) $(NVFLAGS) $(INCLUDES) $(EXTRA) main.o base.o -o $@ $(LIBS)
	mv $@ $(BIN)

$(EXE)_linear: main.o linear.o
	$(NVCC) $(NVFLAGS) $(INCLUDES) $(EXTRA) main.o linear.o -o $@ $(LIBS)
	mv $@ $(BIN)

$(EXE)_topo: main.o topo.o
	$(NVCC) $(NVFLAGS) $(INCLUDES) $(EXTRA) main.o topo.o -o $@ $(LIBS)
	mv $@ $(BIN)

clean:
	rm *.o

