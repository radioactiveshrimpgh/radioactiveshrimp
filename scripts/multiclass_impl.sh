#!/bin/bash

KEYWORD='hw02'

#invoke multiclass_impl.py 5 times with keyword hw02
./multiclass_impl.py -k $KEYWORD
./multiclass_impl.py -k $KEYWORD
./multiclass_impl.py -k $KEYWORD
./multiclass_impl.py -k $KEYWORD
./multiclass_impl.py -k $KEYWORD

# aggregate the 5 files and generate boxplot
./multiclass_eval.py -k $KEYWORD