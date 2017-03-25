HOME=/home/cxh
CUDA_HOME=/usr/local/cuda
GARDINIA_ROOT=$(HOME)/gardinia_code
CC=gcc
CXX=g++
NVCC=$(CUDA_HOME)/bin/nvcc
COMPUTECAPABILITY=sm_20
CXXFLAGS = -O3 -Wall
#CXXFLAGS = -g
PARFLAG = -fopenmp
#NVFLAGS=-g -arch=$(COMPUTECAPABILITY) #-Xptxas -v
NVFLAGS=-O3 -arch=$(COMPUTECAPABILITY) #-Xptxas -v
CUB_DIR=$(HOME)/cub-1.6.4
B40_DIR=$(HOME)/back40computing-read-only
INCLUDES=-I$(CUDA_HOME)/include -I$(GARDINIA_ROOT)/include
LIBS=-L$(CUDA_HOME)/lib64
#EXTRA=-cudart shared
BIN=$(GARDINIA_ROOT)/bin
