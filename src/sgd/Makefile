include ../common.mk
SRC=main.cc
OBJS=base.o

all: $(OBJS) sgd_base

%.o: %.cu
	$(NVCC) $(NVFLAGS) $(INCLUDES) -c $<

sgd_base:
	$(NVCC) $(NVFLAGS) $(INCLUDES) $(EXTRA) $(SRC) base.o -o $@ $(LIBS)
	mv $@ $(BIN)

sgd_lb:
	$(NVCC) $(NVFLAGS) $(INCLUDES) $(EXTRA) $(SRC) lb.o -o $@ $(LIBS)
	mv $@ $(BIN)

clean:
	rm *.o
