OUTPUTDIR := bin/

#-std=c++14

CFLAGS := -std=c++11 -fvisibility=hidden -lpthread -O3
CFLAGS_DBG := -std=c++11 -fvisibility=hidden -lpthread -O3 -DEBUG=1

SOURCES := src/SPDT_squential/*.cpp
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

$(TARGETBIN_DBG): $(SOURCES) $(HEADERS)
	$(CXX) -o $@ $(CFLAGS_DBG) $(SOURCES)

clean:
	rm -rf ./$(TARGETBIN)
	rm -rf ./$(TARGETBIN_DBG)

FILES = src/*.cpp \
		src/*.h