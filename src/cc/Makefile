include ../common.mk
NVFLAGS+=-std=c++11
SRC=main.cc
OBJS=base.o

all: $(OBJS) cc_base

%.o: %.cu
	$(NVCC) $(NVFLAGS) $(INCLUDES) -c $<

cc_base:
	$(NVCC) $(NVFLAGS) $(INCLUDES) $(EXTRA) $(SRC) base.o -o $@ $(LIBS)
	mv $@ $(BIN)

cc_lb:
	$(NVCC) $(NVFLAGS) $(INCLUDES) $(EXTRA) $(SRC) -o $@ $(LIBS)
	mv $@ $(BIN)

clean:
	rm cc
