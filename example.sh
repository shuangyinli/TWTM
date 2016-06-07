#!/bin/bash

#This is simple example how to use twtm for training and testing.

cd ./src/ 
make clean
echo
make
echo
rm -f ./output/*

echo

time ./twtm est ../demo/twtm.demo.input setting.txt 10 ./output

echo

time ./twtm inf ../demo/twtm.demo.input setting.txt ./output final ./output
