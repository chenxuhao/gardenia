include ../common.mk
INCLUDES+=-I $(CUB_DIR)
SRC=main.cu

all: pr_gather pr_scatter pr_fusion

pr_scatter:
	$(NVCC) $(NVFLAGS) $(INCLUDES) $(EXTRA) $(SRC) -DVARIANT=0 -o $@ $(LIBS)
	mv pr_scatter $(BIN)

pr_gather:
	$(NVCC) $(NVFLAGS) $(INCLUDES) $(EXTRA) $(SRC) -DVARIANT=1 -o $@ $(LIBS)
	mv pr_gather $(BIN)

pr_fusion:
	$(NVCC) $(NVFLAGS) $(INCLUDES) $(EXTRA) $(SRC) -DVARIANT=2 -o $@ $(LIBS)
	mv pr_fusion $(BIN)

clean:
	rm pr_scatter pr_gather pr_fusion
