OUTPUTDIR := bin/

#-std=c++14

COPT = -O1
CFLAGS := -std=c++11 -fvisibility=hidden -lpthread $(COPT)

SOURCES := src/SPDT_general/array.cpp src/SPDT_general/main.cpp src/SPDT_general/parser.cpp

SEQUENTIAL = src/SPDT_sequential/tree.cpp 
FEATURE_PARALLEL = src/SPDT_openmp/tree-feature-parallel.cpp
DATA_PARALLEL = src/SPDT_openmp/tree-data-parallel.cpp
CUDA = src/SPDT_CUDA/tree.cu

HEADERS := src/SPDT_general/array.h src/SPDT_general/parser.h src/SPDT_general/tree.h

TARGETBIN := decision-tree
TARGETBIN_DBG := decision-tree-dbg
TARGETBIN_FEATURE := decision-tree-feature
TARGETBIN_DATA := decision-tree-data
TARGETBIN_CUDA := decision-tree-cuda


# Additional flags used to compile decision-tree-dbg
# You can edit these freely to change how your debug binary compiles.
COPT_DBG = -Og
CFLAGS_DBG = -DDEBUG=1

CXX := g++

# parameters for CUDA
LDFLAGS = -L/usr/local/depot/cuda-8.0/lib64/ -lcudart
NVCC = nvcc
NVCCFLAGS = -O3 -std=c++11 -w -m64 --gpu-architecture compute_35
CXX_CUDA = g++ -m64
CXXFLAGS_CUDA = -O3 -std=c++11 -fvisibility=hidden -lpthread -pg
OBJDIR = objs
OBJDIR_CUDA = $(OBJDIR)/SPDT_CUDA
OBJS_CUDA = $(OBJDIR_CUDA)/tree.o $(OBJDIR_CUDA)/parser.o $(OBJDIR_CUDA)/histogram.o $(OBJDIR_CUDA)/main.o 

.SUFFIXES:
.PHONY: all clean

all: $(TARGETBIN)

$(TARGETBIN): $(SOURCES) $(HEADERS) $(SEQUENTIAL) 
	$(CXX) -o $@ $(CFLAGS) $(SOURCES) $(SEQUENTIAL)

debug: $(TARGETBIN_DBG)

# Debug driver
$(TARGETBIN_DBG): COPT = $(COPT_DBG)
$(TARGETBIN_DBG): CFLAGS += $(CFLAGS_DBG)
$(TARGETBIN_DBG): $(SOURCES) $(HEADERS) $(SEQUENTIAL) 
	$(CXX) -o $@ $(CFLAGS) $(SOURCES) $(SEQUENTIAL) 

feature: $(TARGETBIN_FEATURE)
$(TARGETBIN_FEATURE): $(SOURCES) $(HEADERS) $(FEATURE_PARALLEL)
	$(CXX) -o $@ $(CFLAGS) -fopenmp $(SOURCES) $(FEATURE_PARALLEL) 

data: $(TARGETBIN_DATA)
$(TARGETBIN_DATA): $(SOURCES) $(HEADERS) $(DATA_PARALLEL)
	$(CXX) -o $@ $(CFLAGS) -fopenmp  $(SOURCES) $(DATA_PARALLEL)

dirs:
	mkdir -p $(OBJDIR)/
	mkdir -p $(OBJDIR_CUDA)/

cuda: $(TARGETBIN_CUDA)
$(TARGETBIN_CUDA): dirs $(OBJS_CUDA)
	$(CXX_CUDA) $(CXXFLAGS_CUDA) -o $@ $(OBJS_CUDA) $(LDFLAGS)

$(OBJDIR_CUDA)/%.o: src/SPDT_CUDA/%.cpp
	$(CXX_CUDA) $< $(CXXFLAGS_CUDA) -c -o $@

$(OBJDIR_CUDA)/%.o: src/SPDT_CUDA/%.cu
	$(NVCC) $< $(NVCCFLAGS) -c -o $@ 

clean:
	rm -rf ./$(TARGETBIN)
	rm -rf ./$(TARGETBIN_DBG)
	rm -rf ./$(TARGETBIN_DATA)
	rm -rf ./$(TARGETBIN_FEATURE)
	rm -rf ./$(TARGETBIN_CUDA)
	rm -rf $(OBJDIR)
