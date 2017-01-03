CUDA_HOME=/usr/local/cuda
GARDINIA_ROOT=/home/cxh/gardinia
CUB_DIR=/home/cxh/cub-1.1.1
B40_DIR=~/back40computing-read-only
GCC=gcc
GXX=g++
NVCC=$(CUDA_HOME)/bin/nvcc
COMPUTECAPABILITY=sm_20
CXX_FLAGS += -std=c++11 -O3 -Wall
PAR_FLAG = -fopenmp
#NVFLAGS=-g -arch=$(COMPUTECAPABILITY) #-Xptxas -v
NVFLAGS=-w -O3 -arch=$(COMPUTECAPABILITY) #-Xptxas -v
INCLUDES=-I$(CUDA_HOME)/include -I$(GARDINIA_ROOT)/include
LIBS=-L$(CUDA_HOME)/lib64
#EXTRA=-cudart shared
BIN=$(GARDINIA_ROOT)/bin/
