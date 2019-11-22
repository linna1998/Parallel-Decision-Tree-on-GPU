OUTPUTDIR := bin/

#-std=c++14

COPT = -O3
CFLAGS := -std=c++11 -fvisibility=hidden -lpthread $(COPT)

SOURCES := src/SPDT_squential/*.cpp
SOURCES_OPENMP := src/SPDT_openmp/histogram.cpp \
				  src/SPDT_openmp/parser.cpp \
				  src/SPDT_openmp/main.cpp 

FEATURE_PARALLEL := src/SPDT_openmp/tree-feature-parallel.cpp
NODE_PARALLEL := src/SPDT_openmp/tree-node-parallel.cpp


HEADERS := src/SPDT_squential/*.h
HEADERS_OPENMP := src/SPDT_openmp/*.h

TARGETBIN := decision-tree
TARGETBIN_DBG := decision-tree-dbg
TARGETBIN_FEATURE := decision-tree-feature
TARGETBIN_NODE := decision-tree-node


# Additional flags used to compile decision-tree-dbg
# You can edit these freely to change how your debug binary compiles.
COPT_DBG = -Og
CFLAGS_DBG = -DDEBUG=1

CXX := g++

.SUFFIXES:
.PHONY: all clean

all: $(TARGETBIN)

$(TARGETBIN): $(SOURCES) $(HEADERS)
	$(CXX) -o $@ $(CFLAGS) $(SOURCES)

debug: $(TARGETBIN_DBG)

# Debug driver
$(TARGETBIN_DBG): COPT = $(COPT_DBG)
$(TARGETBIN_DBG): CFLAGS += $(CFLAGS_DBG)
$(TARGETBIN_DBG): $(HEADERS)
	$(CXX) -o $@ $^ $(CFLAGS) $(SOURCES)

feature: $(TARGETBIN_FEATURE)

$(TARGETBIN_FEATURE): $(SOURCES_OPENMP) $(HEADERS_OPENMP) $(FEATURE_PARALLEL)
	$(CXX) -o $@ $(CFLAGS) -fopenmp $(SOURCES_OPENMP) $(FEATURE_PARALLEL)

node: $(TARGETBIN_NODE)

$(TARGETBIN_NODE): $(SOURCES_OPENMP) $(HEADERS_OPENMP) $(NODE_PARALLEL)
	$(CXX) -o $@ $(CFLAGS) -fopenmp $(SOURCES_OPENMP) $(NODE_PARALLEL)


clean:
	rm -rf ./$(TARGETBIN)
	rm -rf ./$(TARGETBIN_DBG)

FILES = src/*.cpp \
		src/*.h