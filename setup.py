#########################################
# setup.py                              #
# Setup for PyBKT                       #
#                                       #
# @author Anirudhan Badrinath           #
# Last edited: 01 April 2020            #
#########################################

import numpy as np, os
import sys
from shutil import copyfile, move
import subprocess as s
from sysconfig import get_paths
from distutils.core import setup, Extension

sys.tracebacklimit = 0

FILES = {'synthetic_data_helper.cpp': 'source-cpp/pyBKT/generate/',
         'predict_onestep_states.cpp': 'source-cpp/pyBKT/fit/', 
         'E_step.cpp': 'source-cpp/pyBKT/fit/'}

ALL_COMPILE_ARGS = ['-c', '-fPIC', '-w', '-fopenmp']
ALL_LINK_ARGS = ['-fopenmp']
ALL_LIBRARIES = ['crypt', 'pthread', 'dl', 'util', 'm']
INCLUDE_DIRS = sys.path + [np.get_include(), 'source-cpp/pyBKT/Eigen/', get_paths()['include']]
LIBRARY_DIRS = [os.environ['LD_LIBRARY_PATH']] if 'LD_LIBRARY_PATH' in os.environ \
                                               else []
def find_library_dirs():
    lst = []
    os.system("whereis libboost_python | cut -d' ' -f 2 | sed 's/libboost.*//' > np-include.info")
    lst.append(open("np-include.info", "r").read().strip())
    os.system("python3-config --exec-prefix > np-include.info")
    lst.append(open("np-include.info", "r").read().strip() + "/lib")
    return lst

def find_dep_lib_dirs():
    lst = []
    os.system("ldconfig -p | grep libboost_python | sort -r | head -n1 | cut -d\">\" -f2 | xargs | sed 's/libboost.*//' > np-include.info")
    lst.append(open("np-include.info", "r").read().strip())
    os.system("python3-config --exec-prefix > np-include.info")
    lst.append(open("np-include.info", "r").read().strip() + "/lib")
    return lst

def find_dep_lib_name(l = None):
    if l is None:
        os.system("ldconfig -p | grep libboost_python | sort -r | head -n1 | cut -d'>' -f1 | xargs | sed 's/.so.*//' | sed 's/.*lib//' > np-include.info")
    else:
        os.system("ls " + l + "/libboost_pytho* | sort -r | head -n1 | cut -d'>' -f1 | xargs | sed 's/.so.*//' | sed 's/.*lib//' > np-include.info")
    return open("np-include.info", "r").read().strip()

def find_boost_version():
    os.system("cat $(whereis boost | awk '{print $2}')/version.hpp | grep \"#define BOOST_LIB_VERSION\" | awk '{print $3}' | sed 's\\\"\\\\g' > np-include.info")
    return int(open("np-include.info", "r").read().strip().replace('_', ''))

def copy_files(l, s):
    for i in l:
        copyfile(os.path.normpath(s + "/" + i), os.path.normpath(l[i] + "/" + i))

def clean():
    global LIBRARY_DIRS, ALL_LIBRARIES
    os.remove('np-include.info')
    LIBRARY_DIRS = [i for i in LIBRARY_DIRS if i != ""]
    ALL_LIBRARIES = [i for i in ALL_LIBRARIES if i != ""]


try:
    if LIBRARY_DIRS:
        ALL_LIBRARIES.append(find_dep_lib_name(os.environ['LD_LIBRARY_PATH']))

    if find_boost_version() < 165:
        copy_files(FILES, 'source-cpp/.DEPRECATED')
        LIBRARY_DIRS += find_dep_lib_dirs()
        ALL_LIBRARIES.append(find_dep_lib_name())
    else:
        copy_files(FILES, 'source-cpp/.NEW')
        LIBRARY_DIRS += find_library_dirs()
        ALL_LIBRARIES += ['boost_python3', 'boost_numpy3']

    clean()

    module1 = Extension('pyBKT/generate/synthetic_data_helper',
                        sources = ['source-cpp/pyBKT/generate/synthetic_data_helper.cpp'], 
                        include_dirs = INCLUDE_DIRS,
                        extra_compile_args = ALL_COMPILE_ARGS,
                        library_dirs = LIBRARY_DIRS, 
                        libraries = ALL_LIBRARIES, 
                        extra_link_args = ALL_LINK_ARGS)

    module2 = Extension('pyBKT/fit/E_step', 
                        sources = ['source-cpp/pyBKT/fit/E_step.cpp'],
                        include_dirs = INCLUDE_DIRS,
                        extra_compile_args = ALL_COMPILE_ARGS,
                        library_dirs = LIBRARY_DIRS, 
                        libraries = ALL_LIBRARIES, 
                        extra_link_args = ALL_LINK_ARGS)

    module3 = Extension('pyBKT/fit/predict_onestep_states',
                        sources = ['source-cpp/pyBKT/fit/predict_onestep_states.cpp'],                        include_dirs = INCLUDE_DIRS,
                        extra_compile_args = ALL_COMPILE_ARGS,
                        library_dirs = LIBRARY_DIRS, 
                        libraries = ALL_LIBRARIES, 
                        extra_link_args = ALL_LINK_ARGS)

    setup(
        name="pyBKT",
        version="1.3",
        author="Zachary Pardos, Anirudhan Badrinath, Matthew Jade Johnson, Christian Garay",
        author_email="zp@berkeley.edu, abadrinath@berkeley.edu, mattjj@csail.mit.edu, c.garay@berkeley.edu",
        description="PyBKT",
        url="https://github.com/CAHLR/pyBKT",
        packages=['pyBKT', 'pyBKT.generate', 'pyBKT.fit', 'pyBKT.util'],
        package_dir = { 'pyBKT': 'source-cpp/pyBKT',
                        'pyBKT.generate': 'source-cpp/pyBKT/generate',
                        'pyBKT.fit': 'source-cpp/pyBKT/fit',
                        'pyBKT.util': 'source-cpp/pyBKT/util'},
        classifiers=[
            "Programming Language :: Python :: 3",
            "License :: OSI Approved :: MIT License",
            "Operating System :: OS Independent",
        ],
        install_requires = ["numpy"],
        ext_modules = [module1, module2, module3]
    )
except:
    setup(
        name="pyBKT",
        version="1.3",
        author="Zachary Pardos, Anirudhan Badrinath, Matthew Jade Johnson, Christian Garay",
        author_email="zp@berkeley.edu, abadrinath@berkeley.edu, mattjj@csail.mit.edu, c.garay@berkeley.edu",
        description="PyBKT",
        url="https://github.com/CAHLR/pyBKT",
        classifiers=[
            "Programming Language :: Python :: 3",
            "License :: OSI Approved :: MIT License",
            "Operating System :: OS Independent",
        ],
        packages=['pyBKT', 'pyBKT.generate', 'pyBKT.fit', 'pyBKT.util'],
        package_dir = { 'pyBKT': 'source-py/pyBKT',
                        'pyBKT.generate': 'source-py/pyBKT/generate',
                        'pyBKT.fit': 'source-py/pyBKT/fit',
                        'pyBKT.util': 'source-py/pyBKT/util'},
        install_requires = ["numpy"],
    )
