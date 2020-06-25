#########################################
# setup.py                              #
# Setup for PyBKT                       #
#                                       #
# @author Anirudhan Badrinath           #
# Last edited: 01 April 2020            #
#########################################

import numpy as np, os
from os.path import normpath as npath
import sys
from shutil import copyfile, move
import subprocess as s
from sysconfig import get_paths
from setuptools import setup, Extension

sys.tracebacklimit = 0

FILES = {'synthetic_data_helper.cpp': 'source-cpp/pyBKT/generate/',
         'predict_onestep_states.cpp': 'source-cpp/pyBKT/fit/', 
         'E_step.cpp': 'source-cpp/pyBKT/fit/'}

ALL_COMPILE_ARGS = ['-c', '-fPIC', '-w', '-fopenmp']
ALL_LINK_ARGS = ['-fopenmp']
ALL_LIBRARIES = ['crypt', 'pthread', 'dl', 'util', 'm']
INCLUDE_DIRS = sys.path + [np.get_include(), 'source-cpp/pyBKT/Eigen/', get_paths()['include']] + \
                ([os.environ['BOOST_INCLUDE']] if 'BOOST_INCLUDE' in os.environ \
                                                 else [])
LIBRARY_DIRS = [os.environ['LD_LIBRARY_PATH']] if 'LD_LIBRARY_PATH' in os.environ \
                                               else []
def find_library_dirs():
    lst = []
    try:
        os.system("whereis libboost_python | cut -d' ' -f 2 | sed 's/libboost.*//' > np-include.info")
        lst.append(open("np-include.info", "r").read().strip())
    except:
        pass
    os.system("python3-config --exec-prefix > np-include.info")
    lst.append(open("np-include.info", "r").read().strip() + "/lib")
    return lst

def find_dep_lib_dirs():
    lst = []
    try:
        os.system("ldconfig -p | grep libboost_python | sort -r | head -n1 | cut -d\">\" -f2 | xargs | sed 's/libboost.*//' > np-include.info")
        lst.append(open("np-include.info", "r").read().strip())
    except:
        pass
    os.system("python3-config --exec-prefix > np-include.info")
    lst.append(open("np-include.info", "r").read().strip() + "/lib")
    return lst

def find_dep_lib_name(l = None):
    try:
        if l is None:
            os.system("ldconfig -p | grep libboost_python | sort -r | head -n1 | cut -d'>' -f1 | xargs | sed 's/.so.*//' | sed 's/.*lib//' > np-include.info")
        else:
            os.system("ls " + l + "/libboost_pytho* | sort -r | head -n1 | cut -d'>' -f1 | xargs | sed 's/.so.*//' | sed 's/.*lib//' > np-include.info")
        return open("np-include.info", "r").read().strip()
    except:
        return "boost_python"

def find_boost_version():
    try:
        os.system("cat $(whereis boost | awk '{print $2}')/version.hpp | grep \"#define BOOST_LIB_VERSION\" | awk '{print $3}' | sed 's\\\"\\\\g' > np-include.info")
        return int(open("np-include.info", "r").read().strip().replace('_', ''))
    except:
        return 165

def copy_files(l, s):
    for i in l:
        copyfile(npath(s + "/" + i), npath(l[i] + "/" + i))

def clean():
    global LIBRARY_DIRS, ALL_LIBRARIES
    os.remove('np-include.info')
    LIBRARY_DIRS = [i for i in LIBRARY_DIRS if i != ""]
    ALL_LIBRARIES = [i for i in ALL_LIBRARIES if i != ""]

with open('README.md', encoding='utf-8') as f:
    long_description = f.read()

try:
    if LIBRARY_DIRS:
        ALL_LIBRARIES.append(find_dep_lib_name(os.environ['LD_LIBRARY_PATH']))

    if find_boost_version() < 165:
        copy_files(FILES, npath('source-cpp/.DEPRECATED'))
        LIBRARY_DIRS += find_dep_lib_dirs()
        ALL_LIBRARIES.append(find_dep_lib_name())
    else:
        copy_files(FILES, npath('source-cpp/.NEW'))
        LIBRARY_DIRS += find_library_dirs()
        ALL_LIBRARIES += ['boost_python3', 'boost_numpy3']

    clean()

    module1 = Extension('pyBKT/generate/synthetic_data_helper',
                        sources = [npath('source-cpp/pyBKT/generate/synthetic_data_helper.cpp')], 
                        include_dirs = INCLUDE_DIRS,
                        extra_compile_args = ALL_COMPILE_ARGS,
                        library_dirs = LIBRARY_DIRS, 
                        libraries = ALL_LIBRARIES, 
                        extra_link_args = ALL_LINK_ARGS)

    module2 = Extension('pyBKT/fit/E_step', 
                        sources = [npath('source-cpp/pyBKT/fit/E_step.cpp')],
                        include_dirs = INCLUDE_DIRS,
                        extra_compile_args = ALL_COMPILE_ARGS,
                        library_dirs = LIBRARY_DIRS, 
                        libraries = ALL_LIBRARIES, 
                        extra_link_args = ALL_LINK_ARGS)

    module3 = Extension('pyBKT/fit/predict_onestep_states',
                        sources = [npath('source-cpp/pyBKT/fit/predict_onestep_states.cpp')],
                        include_dirs = INCLUDE_DIRS,
                        extra_compile_args = ALL_COMPILE_ARGS,
                        library_dirs = LIBRARY_DIRS, 
                        libraries = ALL_LIBRARIES, 
                        extra_link_args = ALL_LINK_ARGS)

    setup(
        name="pyBKT",
        version="1.0.1",
        author="Zachary Pardos, Anirudhan Badrinath, Matthew Jade Johnson, Christian Garay",
        author_email="zp@berkeley.edu, abadrinath@berkeley.edu, mattjj@csail.mit.edu, c.garay@berkeley.edu",
        license = 'MIT',
        description="PyBKT - Python Implentation of Bayesian Knowledge Tracing",
        url="https://github.com/CAHLR/pyBKT",
        download_url = 'https://github.com/CAHLR/pyBKT/archive/1.0.tar.gz',
        keywords = ['BKT', 'Bayesian Knowledge Tracing', 'Bayesian Network', 'Hidden Markov Model', 'Intelligent Tutoring Systems', 'Adaptive Learning'],
        classifiers=[
            'Programming Language :: Python :: 3.5',
            'Programming Language :: Python :: 3.6',
            'Programming Language :: Python :: 3.7',
            'Programming Language :: Python :: 3.8',
            "License :: OSI Approved :: MIT License",
            "Operating System :: OS Independent",
        ],
        long_description = long_description,
        long_description_content_type='text/markdown',
        packages=['pyBKT', 'pyBKT.generate', 'pyBKT.fit', 'pyBKT.util'],
        package_dir = { 'pyBKT': npath('source-cpp/pyBKT'),
                        'pyBKT.generate': npath('source-cpp/pyBKT/generate'),
                        'pyBKT.fit': npath('source-cpp/pyBKT/fit'),
                        'pyBKT.util': npath('source-cpp/pyBKT/util')},
        install_requires = ["numpy"],
        ext_modules = [module1, module2, module3]
    )
except:
    setup(
        name="pyBKT",
        version="1.0.1",
        author="Zachary Pardos, Anirudhan Badrinath, Matthew Jade Johnson, Christian Garay",
        author_email="zp@berkeley.edu, abadrinath@berkeley.edu, mattjj@csail.mit.edu, c.garay@berkeley.edu",
        license = 'MIT',
        description="PyBKT - Python Implentation of Bayesian Knowledge Tracing",
        url="https://github.com/CAHLR/pyBKT",
        download_url = 'https://github.com/CAHLR/pyBKT/archive/1.0.tar.gz',
        keywords = ['BKT', 'Bayesian Knowledge Tracing', 'Bayesian Network', 'Hidden Markov Model', 'Intelligent Tutoring Systems', 'Adaptive Learning'],
        classifiers=[
            'Programming Language :: Python :: 3.5',
            'Programming Language :: Python :: 3.6',
            'Programming Language :: Python :: 3.7',
            'Programming Language :: Python :: 3.8',
            "License :: OSI Approved :: MIT License",
            "Operating System :: OS Independent",
        ],
        long_description = long_description,
        long_description_content_type='text/markdown',
        packages=['pyBKT', 'pyBKT.generate', 'pyBKT.fit', 'pyBKT.util'],
        package_dir = { 'pyBKT': npath('source-py/pyBKT'),
                        'pyBKT.generate': npath('source-py/pyBKT/generate'),
                        'pyBKT.fit': npath('source-py/pyBKT/fit'),
                        'pyBKT.util': npath('source-py/pyBKT/util')},
        install_requires = ["numpy"],
    )
