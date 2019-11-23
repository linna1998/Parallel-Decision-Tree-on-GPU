OUTPUTDIR := bin/

#-std=c++14

COPT = -O3
CFLAGS := -std=c++11 -fvisibility=hidden -lpthread $(COPT)

SOURCES := src/SPDT_general/histogram.cpp src/SPDT_general/main.cpp src/SPDT_general/parser.cpp

SEQUENTIAL = src/SPDT_sequential/tree.cpp 
FEATURE_PARALLEL = src/SPDT_openmp/tree-feature-parallel.cpp
DATA_PARALLEL = src/SPDT_openmp/tree-data-parallel.cpp

HEADERS := src/SPDT_general/histogram.h src/SPDT_general/parser.h src/SPDT_general/tree.h

TARGETBIN := decision-tree
TARGETBIN_DBG := decision-tree-dbg
TARGETBIN_FEATURE := decision-tree-feature
TARGETBIN_DATA := decision-tree-data


# Additional flags used to compile decision-tree-dbg
# You can edit these freely to change how your debug binary compiles.
COPT_DBG = -Og
CFLAGS_DBG = -DDEBUG=1

CXX := g++

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


clean:
	rm -rf ./$(TARGETBIN)
	rm -rf ./$(TARGETBIN_DBG)
	rm -rf ./$(TARGETBIN_DATA)
	rm -rf ./$(TARGETBIN_FEATURE)
