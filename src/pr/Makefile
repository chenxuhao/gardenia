include ../common.mk
INCLUDES+=-I $(CUB_DIR)
SRC=main.cu
OBJS=gather.o scatter.o fusion.o

all: $(OBJS) pr_gather pr_scatter pr_fusion

%.o: %.cu
	$(NVCC) $(NVFLAGS) $(INCLUDES) -c $<

pr_scatter:
	$(NVCC) $(NVFLAGS) $(INCLUDES) $(EXTRA) $(SRC) scatter.o -o $@ $(LIBS)
	mv $@ $(BIN)

pr_gather:
	$(NVCC) $(NVFLAGS) $(INCLUDES) $(EXTRA) $(SRC) gather.o -o $@ $(LIBS)
	mv $@ $(BIN)

pr_fusion:
	$(NVCC) $(NVFLAGS) $(INCLUDES) $(EXTRA) $(SRC) fusion.o -o $@ $(LIBS)
	mv $@ $(BIN)

clean:
	rm pr_scatter pr_gather pr_fusion
