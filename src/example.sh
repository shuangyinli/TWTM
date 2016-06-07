#!/bin/bash

#This is simple example how to use CTL for training and testing.

#The train set is a very small part of training set with 1,000 documents, and 200 documents are for testing.
#
#Check ../demo/ to show the input files: training set, test set.
#Check ./output to show the output 

make clean
echo
make
echo
rm -f ./output/*

echo

time ./ctl est ../demo/train_wiki1000.txt setting.txt 10 ./output

echo

time ./ctl inf ../demo/test setting.txt N ./output final ./input/test_init_topics_doc ./output
