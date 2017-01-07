include ../common.mk
INCLUDES+=-I $(CUB_DIR)
SRC=main.cc
OBJS=base.o topo.o bitset.o

all: $(OBJS) color_base color_bitset color_topo

%.o: %.cu
	$(NVCC) $(NVFLAGS) $(INCLUDES) -c $<

color_topo:
	$(NVCC) $(NVFLAGS) $(INCLUDES) $(EXTRA) $(SRC) topo.o -o $@
	mv $@ $(BIN)

color_base:
	$(NVCC) $(NVFLAGS) $(INCLUDES) $(EXTRA) $(SRC) base.o -o $@
	mv $@ $(BIN)

color_bitset:
	$(NVCC) $(NVFLAGS) $(INCLUDES) $(EXTRA) $(SRC) bitset.o -o $@
	mv $@ $(BIN)

clean:
	rm *.o

