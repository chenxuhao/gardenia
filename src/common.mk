CUDA_HOME=/usr/local/cuda
GARDINIA_ROOT=/Users/cxh/Work/gardinia
GCC=gcc
GXX=g++
NVCC=$(CUDA_HOME)/bin/nvcc
COMPUTECAPABILITY=sm_35
CXX_FLAGS += -std=c++11 -O3 -Wall
PAR_FLAG = -fopenmp
#NVFLAGS=-g -arch=$(COMPUTECAPABILITY) #-Xptxas -v
NVFLAGS=-w -O3 -arch=$(COMPUTECAPABILITY) #-Xptxas -v
INCLUDES=-I$(CUDA_HOME)/include -I$(GARDINIA_ROOT)/include
BIN=$(GARDINIA_ROOT)/bin/
