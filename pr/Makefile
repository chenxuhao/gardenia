include ../common.mk
INCLUDES+=-I ~/cub-1.1.1
LIBS=-L$(CUDA_HOME)/lib64
EXTRA=-cudart shared
SRC=main.cu

all: pr_scatter pr_gather pr_lb

pr_scatter:
	$(NVCC) $(NVFLAGS) $(INCLUDES) $(EXTRA) $(SRC) -DVARIANT=0 -o $@ $(LIBS)
	mv pr_scatter $(BIN)

pr_gather:
	$(NVCC) $(NVFLAGS) $(INCLUDES) $(EXTRA) $(SRC) -DVARIANT=1 -o $@ $(LIBS)
	mv pr_gather $(BIN)

pr_lb:
	$(NVCC) $(NVFLAGS) $(INCLUDES) $(EXTRA) $(SRC) -DVARIANT=2 -o $@ $(LIBS)
	mv pr_lb $(BIN)

clean:
	rm pr_scatter pr_gather pr_lb
