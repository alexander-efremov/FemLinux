# hello from MAC

# Location of the CUDA Toolkit
CUDA_PATH  ?= /usr/local/cuda-6.0
NVCC := $(CUDA_PATH)/bin/nvcc -ccbin g++

# internal flags
NVCCFLAGS   := -m64 -O2
CCFLAGS     := -m64 -O2
LDFLAGS     := 


ifeq ($(flagman), 1)
   NVCCFALGS += -DFLAGMAN
endif

# Extra user flags
EXTRA_NVCCFLAGS   ?=
EXTRA_LDFLAGS     ?= -L./lib
EXTRA_CCFLAGS     ?=

# Debug build flags
ifeq ($(dbg), 1)
      NVCCFLAGS += -g -G -DDEBUG 
endif

ALL_CCFLAGS :=
ALL_CCFLAGS += $(NVCCFLAGS)
ALL_CCFLAGS += $(EXTRA_NVCCFLAGS)
ALL_CCFLAGS += $(addprefix -Xcompiler ,$(CCFLAGS))
ALL_CCFLAGS += $(addprefix -Xcompiler ,$(EXTRA_CCFLAGS))

ALL_LDFLAGS :=
ALL_LDFLAGS += $(ALL_CCFLAGS)
ALL_LDFLAGS += $(addprefix -Xlinker ,$(LDFLAGS))
ALL_LDFLAGS += $(addprefix -Xlinker ,$(EXTRA_LDFLAGS))

INCLUDES  := -I./include -I./include/gtest
LIBRARIES :=
LIBRARIES += -lgtest -lgtest_main

################################################################################

GENCODE_SM20    := -gencode arch=compute_20,code=sm_20
GENCODE_FLAGS   ?= $(GENCODE_SM20)

################################################################################

# Target rules
all: build

build: test_fixture

low_ord_oper.o: src/low_ord_oper.cpp
	$(NVCC) $(INCLUDES) $(ALL_CCFLAGS) $(GENCODE_FLAGS) -o $@ -c $<

model_data_provider.o: src/model_data_provider.cpp
	$(NVCC) $(INCLUDES) $(ALL_CCFLAGS) $(GENCODE_FLAGS) -o $@ -c $<

file_reader.o: src/file_reader.cpp
	$(NVCC) $(INCLUDES) $(ALL_CCFLAGS) $(GENCODE_FLAGS) -o $@ -c $<

correctness_test.o: src/correctness_test.cpp
	$(NVCC) $(INCLUDES) $(ALL_CCFLAGS) $(GENCODE_FLAGS) -o $@ -c $<

low_ord_oper_cuda.o: src/low_ord_oper.cu
	$(NVCC) $(INCLUDES) $(ALL_CCFLAGS) $(GENCODE_FLAGS) -o $@ -c $<

gtest_main.o: src/gtest_main.cc
	$(NVCC) $(INCLUDES) $(ALL_CCFLAGS) $(GENCODE_FLAGS) -o $@ -c $<


test_fixture: gtest_main.o low_ord_oper.o low_ord_oper_cuda.o model_data_provider.o file_reader.o correctness_test.o
	$(NVCC) $(ALL_LDFLAGS) $(GENCODE_FLAGS) -o $@ $+ $(LIBRARIES)

run: build
	./test_fixture

clean:
	rm -f test_fixture get_quad_coord.o gtest_main.o low_ord_oper.o low_ord_oper_cuda.o model_data_provider.o file_reader.o correctness_test.o

clobber: clean
