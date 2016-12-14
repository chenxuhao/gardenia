include ../common.mk
#INCLUDES+=-I ~/cub-1.1.1
LIBS=-L$(CUDA_HOME)/lib64
#EXTRA=-cudart shared
SRC=main.cu

all: bc

bc:
	$(NVCC) $(NVFLAGS) $(INCLUDES) $(EXTRA) $(SRC) -DVARIANT=0 -o $@ $(LIBS)
	mv bc $(BIN)

bc_lb:
	$(NVCC) $(NVFLAGS) $(INCLUDES) $(EXTRA) $(SRC) -DVARIANT=1 -o $@ $(LIBS)
	mv bc_lb $(BIN)

clean:
	rm bc
