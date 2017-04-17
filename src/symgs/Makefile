include ../common.mk
CXXFLAGS+=-std=c++11
SRC=main.cc
OBJS=base.o verifier.o main.o

all: $(OBJS) symgs_base symgs_omp symgs_vector

%.o: %.cc
	$(CXX) $(CXXFLAGS) $(PARFLAG) $(INCLUDES) -c $<

%.o: %.cu
	$(NVCC) $(NVFLAGS) $(INCLUDES) -c $<

symgs_omp: omp_base.o main.o verifier.o ../vc/omp_base.o
	$(CXX) $(CXXFLAGS) $(INCLUDES) main.o omp_base.o verifier.o ../vc/omp_base.o -o $@ -lgomp
	mv $@ $(BIN)

symgs_base: main.o base.o verifier.o ../vc/omp_base.o
	$(NVCC) $(NVFLAGS) $(INCLUDES) $(EXTRA) main.o base.o verifier.o ../vc/omp_base.o -o $@ $(LIBS) -lgomp
	mv $@ $(BIN)

symgs_vector: main.o vector.o verifier.o ../vc/omp_base.o
	$(NVCC) $(NVFLAGS) $(INCLUDES) $(EXTRA) main.o vector.o verifier.o ../vc/omp_base.o -o $@ $(LIBS) -lgomp
	mv $@ $(BIN)

clean:
	rm *.o
