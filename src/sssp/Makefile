include ../common.mk
INCLUDES+=-I $(CUB_DIR)
EXE=sssp
SRC=main.cc
OBJS= topo.o base.o linear.o

all: $(OBJS) $(EXE)_base $(EXE)_linear $(EXE)_topo

%.o: %.cc
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $<

%.o: %.cu
	$(NVCC) $(NVFLAGS) $(INCLUDES) -c $<

$(EXE)_base:
	$(NVCC) $(NVFLAGS) $(INCLUDES) $(EXTRA) $(SRC) base.o -o $@ $(LIBS)
	mv $@ $(BIN)

$(EXE)_linear:
	$(NVCC) $(NVFLAGS) $(INCLUDES) $(EXTRA) $(SRC) linear.o -o $@ $(LIBS)
	mv $@ $(BIN)

$(EXE)_topo:
	$(NVCC) $(NVFLAGS) $(INCLUDES) $(EXTRA) $(SRC) topo.o -o $@ $(LIBS)
	mv $@ $(BIN)

clean:
	rm *.o

