OUTPUTDIR := bin/

#-std=c++14

COPT = -O3
CFLAGS := -std=c++11 -fvisibility=hidden -lpthread -pg -fopenmp $(COPT)

SOURCES := src/SPDT_squential/histogram.cpp src/SPDT_squential/main.cpp src/SPDT_squential/parser.cpp src/SPDT_squential/tree.cpp

HEADERS := src/SPDT_squential/*.h

TARGETBIN := decision-tree
TARGETBIN_DBG := decision-tree-dbg

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

clean:
	rm -rf ./$(TARGETBIN)
	rm -rf ./$(TARGETBIN_DBG)

FILES = src/*.cpp \
		src/*.h