#!/bin/bash

conda_path=$(conda info --envs | grep base | awk 'NF>1{print $NF}')
conda env create -f environment.yml
export BOOST_INCLUDE="$conda_path/envs/pyBKT/include" && export LD_LIBRARY_PATH="$conda_path/envs/pyBKT/lib" && conda activate pyBKT && pip install pyBKT
echo "Done installing pyBKT!"
