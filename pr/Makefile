include ../common.mk
INCLUDES+=-I ~/cub-1.1.1
LIBS=-L$(CUDA_HOME)/lib64
EXTRA=-cudart shared
SRC=main.cu

all: pr_wlw pr_wlc

pr_wlw:
	$(NVCC) $(NVFLAGS) $(INCLUDES) $(EXTRA) $(SRC) -DVARIANT=1 -o $@ $(LIBS)
	mv pr_wlw $(BIN)

pr_wlc:
	$(NVCC) $(NVFLAGS) $(INCLUDES) $(EXTRA) $(SRC) -DVARIANT=1 -o $@ $(LIBS)
	mv pr_wlc $(BIN)

clean:
	rm pr_wlw pr_wlc
