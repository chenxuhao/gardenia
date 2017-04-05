include ../common.mk
SRC=main.cc
OBJS=base.o verifier.o

all: $(OBJS) sgd_base

%.o: %.cc sgd.h
	$(CXX) $(CXXFLAGS) $(PARFLAG) $(INCLUDES) -c $<

%.o: %.cu sgd.h
	$(NVCC) $(NVFLAGS) $(INCLUDES) -c $<

sgd_base: base.o verifier.o ../../include/graph_io.h sgd.h
	$(NVCC) $(NVFLAGS) $(INCLUDES) $(EXTRA) $(SRC) base.o verifier.o -o $@ $(LIBS)
	mv $@ $(BIN)

sgd_lb:
	$(NVCC) $(NVFLAGS) $(INCLUDES) $(EXTRA) $(SRC) lb.o -o $@ $(LIBS)
	mv $@ $(BIN)

clean:
	rm *.o
