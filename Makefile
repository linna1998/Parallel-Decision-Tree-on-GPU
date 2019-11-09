OUTPUTDIR := bin/

#-std=c++14

CFLAGS := -std=c++11 -fvisibility=hidden -lpthread -O3

SOURCES := src/SPDT_squential/histogram.cpp src/SPDT_squential/main.cpp src/SPDT_squential/parser.cpp src/SPDT_squential/tree.cpp
SOURCES_dbg := src/SPDT_squential/histogram.cpp src/SPDT_squential/main_dbg.cpp src/SPDT_squential/parser.cpp src/SPDT_squential/tree.cpp

HEADERS := src/SPDT_squential/*.h

TARGETBIN := decision-tree
TARGETBIN_DBG := decision-tree-dgb

CXX := g++

.SUFFIXES:
.PHONY: all clean

all: $(TARGETBIN)

$(TARGETBIN): $(SOURCES) $(HEADERS)
	$(CXX) -o $@ $(CFLAGS) $(SOURCES)

debug: $(TARGETBIN_DBG)

$(TARGETBIN_DBG): $(SOURCES_dbg) $(HEADERS)
	$(CXX) -o $@ $(CFLAGS) $(SOURCES_dbg)

clean:
	rm -rf ./$(TARGETBIN)
	rm -rf ./$(TARGETBIN_DBG)

FILES = src/*.cpp \
		src/*.h