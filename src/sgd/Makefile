include ../common.mk
SRC=main.cu

all: sgd

sgd:
	$(NVCC) $(NVFLAGS) $(INCLUDES) $(EXTRA) $(SRC) -DVARIANT=0 -o $@ $(LIBS)
	mv sgd $(BIN)

sgd_lb:
	$(NVCC) $(NVFLAGS) $(INCLUDES) $(EXTRA) $(SRC) -DVARIANT=1 -o $@ $(LIBS)
	mv sgd_lb $(BIN)

clean:
	rm sgd
