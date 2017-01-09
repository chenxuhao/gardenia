include ../common.mk
CXXFLAGS+=-std=c++11
SRC=main.cc
OBJS=base.o main.o

all: $(OBJS) cc_base

%.o: %.cc
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $<

%.o: %.cu
	$(NVCC) $(NVFLAGS) $(INCLUDES) -c $<

cc_base: main.o base.o
	$(NVCC) $(NVFLAGS) $(INCLUDES) $(EXTRA) main.o base.o -o $@ $(LIBS)
	mv $@ $(BIN)

cc_lb:
	$(NVCC) $(NVFLAGS) $(INCLUDES) $(EXTRA) main.o -o $@ $(LIBS)
	mv $@ $(BIN)

clean:
	rm *.o
