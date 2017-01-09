include ../common.mk
INCLUDES+=-I $(CUB_DIR)
SRC=main.cc
OBJS=base.o topo.o bitset.o main.o

all: $(OBJS) color_base color_bitset color_topo

%.o: %.cc
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $<

%.o: %.cu
	$(NVCC) $(NVFLAGS) $(INCLUDES) -c $<

color_topo: main.o topo.o
	$(NVCC) $(NVFLAGS) $(INCLUDES) $(EXTRA) main.o topo.o -o $@
	mv $@ $(BIN)

color_base: main.o base.o
	$(NVCC) $(NVFLAGS) $(INCLUDES) $(EXTRA) main.o base.o -o $@
	mv $@ $(BIN)

color_bitset: main.o bitset.o
	$(NVCC) $(NVFLAGS) $(INCLUDES) $(EXTRA) main.o bitset.o -o $@
	mv $@ $(BIN)

clean:
	rm *.o

