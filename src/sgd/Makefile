include ../common.mk
INCLUDES+=-I$(CUB_DIR)
SRC=main.cc
OBJS=base.o vector.o omp_base.o main.o verifier.o
CXXFLAGS+=-std=c++11

all: $(OBJS) sgd_base sgd_omp_base sgd_vector

%.o: %.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $<

%.o: %.cc sgd.h
	$(CXX) $(CXXFLAGS) $(PARFLAG) $(INCLUDES) -c $<

%.o: %.cu sgd.h
	$(NVCC) $(NVFLAGS) $(INCLUDES) -c $<

sgd_base: base.o main.o verifier.o sgd.h
	$(NVCC) $(NVFLAGS) $(INCLUDES) main.o base.o verifier.o -o $@ $(LIBS)
	mv $@ $(BIN)

sgd_vector: vector.o main.o verifier.o sgd.h
	$(NVCC) $(NVFLAGS) $(INCLUDES) main.o vector.o verifier.o -o $@ $(LIBS)
	mv $@ $(BIN)

sgd_omp_base: omp_base.o main.o verifier.o
	$(CXX) $(CXXFLAGS) $(INCLUDES) main.o omp_base.o verifier.o -o $@ -lgomp
	mv $@ $(BIN)

sgd_omp_target: main.o omp_target.o verifier.o
	$(CXX) $(CXXFLAGS) $(INCLUDES) main.o omp_target.o verifier.o -o $@ $(LIBS)
	mv $@ $(BIN)

clean:
	rm *.o
