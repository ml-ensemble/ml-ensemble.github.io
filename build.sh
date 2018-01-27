#! /usr/bin/bash 

cd docs/source 
make docs 

cd ../../info/source
make docs

cd ../..
gulp
