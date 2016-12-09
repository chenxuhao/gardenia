include ../common.mk
#INCLUDES+=-I ~/cub-1.1.1
LIBS=-L$(CUDA_HOME)/lib64
#EXTRA=-cudart shared
SRC=main.cu

all: tc

tc:
	$(NVCC) $(NVFLAGS) $(INCLUDES) $(EXTRA) $(SRC) -DVARIANT=0 -o $@ $(LIBS)
	mv tc $(BIN)

tc_lb:
	$(NVCC) $(NVFLAGS) $(INCLUDES) $(EXTRA) $(SRC) -DVARIANT=1 -o $@ $(LIBS)
	mv tc_lb $(BIN)

clean:
	rm tc
