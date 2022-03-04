include ../common.mk
BLISS=../../lib/bliss-0.73
INCLUDES+=-I$(CUB_DIR) -I./ -I$(BLISS)
CXXFLAGS+=-w -std=c++11
SRC=main.cc
BLISS_OBJS=\
			$(BLISS)/graph.o \
			$(BLISS)/partition.o \
			$(BLISS)/orbit.o
OBJS=serial.o omp_base.o main.o
all: $(OBJS) fsm_serial fsm_omp_base

%.o: %.cpp
	$(ICPC) $(ICPCFLAGS) $(INCLUDES) -c $<

%.o: %.cc
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $<

%.o: %.cu
	$(NVCC) $(NVFLAGS) $(INCLUDES) -c $<

fsm_base: base.o main.o verifier.o
	$(NVCC) $(NVFLAGS) $(INCLUDES) $(EXTRA) main.o base.o verifier.o -o $@ $(LIBS)
	mv $@ $(BIN)

fsm_warp: warp.o main.o verifier.o
	$(NVCC) $(NVFLAGS) $(INCLUDES) $(EXTRA) main.o warp.o verifier.o -o $@ $(LIBS)
	mv $@ $(BIN)

fsm_serial: serial.o main.o
	$(CXX) $(CXXFLAGS) $(INCLUDES) serial.o main.o $(BLISS_OBJS) -o $@
	mv $@ $(BIN)

fsm_omp_base: omp_base.o main.o
	$(CXX) $(CXXFLAGS) $(INCLUDES) omp_base.o main.o $(BLISS_OBJS) -o $@ -lgomp
	mv $@ $(BIN)

fsm_ocl_base: ocl_base.o main.o verifier.o
	$(CXX) $(CXXFLAGS) $(INCLUDES) main.o ocl_base.o verifier.o -o $@ -lOpenCL
	mv $@ $(BIN)

fsm_omp_target: main.o omp_target.o verifier.o
	$(ICPC) $(CXXFLAGS) $(INCLUDES) $(EXTRA) main.o omp_target.o verifier.o -o $@ $(LIBS)
	mv $@ $(BIN)

clean:
	rm *.o
