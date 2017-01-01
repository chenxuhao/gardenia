include ../common.mk
INCLUDES+=-I $(CUB_DIR)
SRC=main.cu

all: color_base color_bitset color_topo

color_topo:
	$(NVCC) $(NVFLAGS) $(INCLUDES) $(EXTRA) $(SRC) -DVARIANT=0 -o $@
	mv $@ $(BIN)

color_base:
	$(NVCC) $(NVFLAGS) $(INCLUDES) $(EXTRA) $(SRC) -DVARIANT=1 -o $@
	mv $@ $(BIN)

color_bitset:
	$(NVCC) $(NVFLAGS) $(INCLUDES) $(EXTRA) $(SRC) -DVARIANT=2 -o $@
	mv $@ $(BIN)

clean:
	rm color_base color_bitset color_topo

