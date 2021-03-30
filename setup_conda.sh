#!/bin/bash

read -e -p "Please enter the location of the Conda install (no relative paths like .. or ~; absolute path only); tab to complete: " conda_path
conda env create -f environment.yml
export BOOST_INCLUDE="$conda_path/envs/pyBKT/include" && export LD_LIBRARY_PATH="$conda_path/envs/pyBKT/lib" && conda activate pyBKT && pip install pyBKT
echo "Done installing pyBKT!"
