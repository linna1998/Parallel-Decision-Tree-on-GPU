OUTPUTDIR := bin/

#-std=c++14
CFLAGS := -std=c++11 -fvisibility=hidden -lpthread -O3

SOURCES := src/SPDT_squential/*.cpp
HEADERS := src/SPDT_squential/*.h

TARGETBIN := decision-tree

CXX := g++

.SUFFIXES:
.PHONY: all clean

all: $(TARGETBIN)

$(TARGETBIN): $(SOURCES) $(HEADERS)
	$(CXX) -o $@ $(CFLAGS) $(SOURCES)

clean:
	rm -rf ./$(TARGETBIN)

FILES = src/*.cpp \
		src/*.h