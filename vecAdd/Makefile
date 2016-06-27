include ../common.mk
INCLUDES+=-I./cub-1.1.1
LIBS=-L$(CUDA_HOME)/lib64
EXTRA=-cudart shared
SRC=main.cu

all:
	$(NVCC) $(NVFLAGS) $(INCLUDES) $(EXTRA) $(SRC) -o vecAdd $(LIBS)
	mv vecAdd $(BIN)

clean:
	rm vecAdd

