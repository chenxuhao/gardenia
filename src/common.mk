CUDA_HOME=/usr/local/cuda
ICC_HOME=/opt/intel/compilers_and_libraries/linux/bin/intel64
MKLROOT=/opt/intel/mkl
CUB_DIR=../../cub
B40_DIR=../../back40computing-read-only
BIN=../../bin
HOST=X86
ifeq ($(HOST),X86)
CC=gcc
CXX=g++
else 
CC=aarch64-linux-gnu-gcc
CXX=aarch64-linux-gnu-g++
endif
ICC=$(ICC_HOME)/icc
ICPC=$(ICC_HOME)/icpc
NVCC=nvcc
COMPUTECAPABILITY=sm_60
CUDA_ARCH = -gencode arch=compute_35,code=sm_35
CUDA_ARCH += -gencode arch=compute_60,code=sm_60
CXXFLAGS=-Wall -fopenmp
ICPCFLAGS=-O3 -Wall -qopenmp
NVFLAGS=$(CUDA_ARCH)
#NVFLAGS+=-Xptxas -v
#NVFLAGS+=-cudart shared
#SIMFLAGS=-O3 -Wall -DSIM -fopenmp -static -L/home/cxh/m5threads/ -lpthread
SIMFLAGS=-flto -fwhole-program -O3 -Wall -DSIM -fopenmp -static
M5OP=/home/cxh/gem5-ics/util/m5/m5op_arm_A64.S
ifeq ($(DEBUG), 1)
	CXXFLAGS += -g -O0
	NVFLAGS += -G
else
	CXXFLAGS += -O3
	NVFLAGS += -O3
endif
INCLUDES = -I$(CUDA_HOME)/include -I../../include
LIBS = -L$(CUDA_HOME)/lib64
