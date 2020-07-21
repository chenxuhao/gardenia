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
CUDA_ARCH := \
	-gencode arch=compute_37,code=sm_37 \
	-gencode arch=compute_61,code=sm_61
#	-gencode arch=compute_70,code=sm_70
CXXFLAGS=-Wall -fopenmp
ICPCFLAGS=-O3 -Wall -qopenmp
NVFLAGS=$(CUDA_ARCH)
#NVFLAGS+=-Xptxas -v
#DEBUG=1
SIMFLAGS=-flto -fwhole-program -O3 -Wall -DSIM -fopenmp -static
M5OP=/home/cxh/gem5-ics/util/m5/m5op_arm_A64.S
ifeq ($(DEBUG), 1)
	CXXFLAGS += -g -O0
	NVFLAGS += -G -lineinfo
else
	CXXFLAGS += -O3
	NVFLAGS += -O3 -w
endif
INCLUDES = -I../../include
CU_INC = -I/usr/include/cuda
#INCLUDES += $(CU_INC)
LIBS = -L/usr/lib64
