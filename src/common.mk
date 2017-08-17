HOME=/home/cxh
CUDA_HOME=/usr/local/cuda
ICC_HOME=/opt/intel/composer_xe_2015.1.133/bin/intel64
GARDENIA_ROOT=$(HOME)/gardenia_code
CUB_DIR=$(HOME)/cub-1.6.4
B40_DIR=$(HOME)/back40computing-read-only
BIN=$(GARDENIA_ROOT)/bin
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
NVCC=$(CUDA_HOME)/bin/nvcc
COMPUTECAPABILITY=sm_20
CXXFLAGS=-O3 -Wall -fopenmp
ICPCFLAGS=-O3 -Wall -openmp
#SIMFLAGS=-O3 -Wall -DSIM -fopenmp -static -L/home/cxh/m5threads/ -lpthread
SIMFLAGS=-flto -fwhole-program -O3 -Wall -DSIM -fopenmp -static
M5OP=$(HOME)/gem5-ics/util/m5/m5op_arm_A64.S
#NVFLAGS=-g -arch=$(COMPUTECAPABILITY) #-Xptxas -v
NVFLAGS=-O3 -arch=$(COMPUTECAPABILITY) -Wno-deprecated-gpu-targets#-Xptxas -v
#NVFLAGS+=-cudart shared
INCLUDES=-I$(CUDA_HOME)/include -I$(GARDENIA_ROOT)/include
LIBS=-L$(CUDA_HOME)/lib64
