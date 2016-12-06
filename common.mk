CUDA_HOME=/usr/local/cuda
GCC=gcc
GXX=g++
NVCC=$(CUDA_HOME)/bin/nvcc
COMPUTECAPABILITY=sm_35
#NVFLAGS=-g -arch=$(COMPUTECAPABILITY) #-Xptxas -v
NVFLAGS=-w -O3 -arch=$(COMPUTECAPABILITY) #-Xptxas -v
INCLUDES=-I$(CUDA_HOME)/include -I../include
BIN=../bin/
