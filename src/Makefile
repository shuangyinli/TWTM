export CC = gcc
export CXX = g++
export CFLAGS = -w -O3 -pthread

INSTALL_PATH=bin/
BIN = twtm
OBJ = twtm-estimate.o  twtm-inference.o twtm-learn.o
.PHONY: clean all

all: $(BIN)

twtm:twtm.c twtm-estimate.o twtm-inference.o twtm-learn.o twtm.h
twtm-estimate.o:twtm-estimate.c utils.h twtm.h twtm-estimate.h
twtm-inference.o: twtm-inference.c utils.h twtm-inference.h twtm.h
twtm-learn.o: twtm-learn.c utils.h twtm-learn.h twtm.h

$(BIN) : 
	$(CXX) $(CFLAGS) -o $@ $(filter %.cpp %.o %.c, $^)

$(OBJ) :
	$(CXX) -c $(CFLAGS) -o $@ $(filter %.cpp %.c, $^)

install:
	cp -f -r $(BIN) $(INSTALL_PATH)

clean:
	$(RM) $(OBJ) $(BIN) *~
