include ../common.mk
BC := bc

program_CXX_SRCS := $(wildcard *.cpp)
program_CXX_OBJS := ${program_CXX_SRCS:.cpp=.o}
program_CU_SRCS := $(wildcard *.cu)
program_CU_OBJS := ${program_CU_SRCS:.cu=.o}
program_INCLUDE_DIRS := .
program_LIBRARIES := 
CPPFLAGS += $(foreach includedir,$(program_INCLUDE_DIRS),-I$(includedir))
CXXFLAGS += -g -O3 -std=c++0x -Wall -pedantic #-Werror
LDADD += $(foreach library,$(program_LIBRARIES),-l$(library))
#GEN_SM35 := -gencode=arch=compute_35,code=\"sm_35,compute_35\"
GEN_SM20 := -gencode=arch=compute_20,code=\"sm_20,compute_20\"
NVFLAGS := -O3 
NVFLAGS += $(GEN_SM35) $(GEN_SM20)
CXXFLAGS += $(EXTRA_FLAGS)
NVFLAGS += $(EXTRA_FLAGS)
LDFLAGS := -L/usr/lib
EXTRA=-cudart shared

.PHONY: all clean distclean

all: $(BC)

#Note: Adding debugging symbols gives bc_gpu_opt too many registers and causes a "too many resources" error. Only the -G flag disturbs things in this way.
debug: CXXFLAGS = -g -O0 -std=c++0x -Wall -pedantic -DDEBUG $(EXTRA_FLAGS)
debug: NVFLAGS = -g -G -O0 $(GEN_SM35) $(GEN_SM20) -DDEBUG $(EXTRA_FLAGS)
debug: $(BC)

debugv: CXXFLAGS = -g -O0 -std=c++0x -Wall -pedantic -DDEBUG $(EXTRA_FLAGS)
debugv: NVFLAGS = -g -G -O0 --ptxas-options="-v" -DDEBUG $(GEN_SM35) $(GEN_SM20) $(EXTRA_FLAGS)
debugv: $(BC)

$(BC): $(program_CXX_OBJS) $(program_CU_SRCS)
	$(NVCC) $(NVFLAGS) $(CPPFLAGS) $(LDFLAGS) $(EXTRA) $(program_CU_SRCS) $(program_CXX_OBJS) $(LDADD) -o $(BC)
	mv $@ $(BIN)

clean:
	@- $(RM) $(BC) $(program_CXX_OBJS) $(program_CU_OBJS) *~ 

distclean: clean
