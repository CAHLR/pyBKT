import os

conda_path = input("Please enter the location of the Conda install: ")
os.environ['BOOST_INCLUDE'] = os.path.normpath(conda_path + "/include")
os.environ['LD_LIBRARY_PATH'] = os.path.normpath(conda_path + "/lib")
os.system('conda env create -f environment.yml')
