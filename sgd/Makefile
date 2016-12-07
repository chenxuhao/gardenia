include ../common.mk
#INCLUDES+=-I ~/cub-1.1.1
LIBS=-L$(CUDA_HOME)/lib64
#EXTRA=-cudart shared
SRC=main.cu

all: sgd_topo

sgd_topo:
	$(NVCC) $(NVFLAGS) $(INCLUDES) $(EXTRA) $(SRC) -DVARIANT=0 -o $@ $(LIBS)
	mv sgd_topo $(BIN)

sgd_wlw:
	$(NVCC) $(NVFLAGS) $(INCLUDES) $(EXTRA) $(SRC) -DVARIANT=1 -o $@ $(LIBS)
	mv sgd_wlw $(BIN)

sgd_wlc:
	$(NVCC) $(NVFLAGS) $(INCLUDES) $(EXTRA) $(SRC) -DVARIANT=2 -o $@ $(LIBS)
	mv sgd_wlc $(BIN)

clean:
	rm sgd_wlw sgd_wlc sgd_topo
