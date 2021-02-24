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
import platform

sys.tracebacklimit = 0

FILES = {'synthetic_data_helper.cpp': 'source-cpp/pyBKT/generate/',
         'predict_onestep_states.cpp': 'source-cpp/pyBKT/fit/', 
         'E_step.cpp': 'source-cpp/pyBKT/fit/'}

if platform.system() == 'Darwin':
    DYNAMIC_LIB = '.dylib'
    try:
        if 'BOOST_INCLUDE' not in os.environ:
            os.environ['BOOST_INCLUDE'] = '/usr/local/Cellar/boost/' + sorted(os.listdir('/usr/local/Cellar/boost/'))[-1] + '/include'
        if 'LD_LIBRARY_PATH' not in os.environ:
            os.environ['LD_LIBRARY_PATH'] = '/usr/local/Cellar/boost-python3/' + sorted(os.listdir('/usr/local/Cellar/boost-python3/'))[-1] + '/lib'
    except:
        pass
    ALL_COMPILE_ARGS = ['-c', '-fPIC', '-w', '-O3', '-stdlib=libc++', '-Xpreprocessor', '-fopenmp']
    ALL_LINK_ARGS = ['-stdlib=libc++']
    ALL_LIBRARIES = ['pthread', 'dl', 'util', 'm', 'omp']
else:
    DYNAMIC_LIB = '.so'
    ALL_COMPILE_ARGS = ['-c', '-fPIC', '-w', '-fopenmp', '-O2']
    ALL_LINK_ARGS = ['-fopenmp']
    ALL_LIBRARIES = ['pthread', 'dl', 'util', 'm']
INCLUDE_DIRS = sys.path + [np.get_include(), 'source-cpp/pyBKT/Eigen/', get_paths()['include']] + \
                ([os.environ['BOOST_INCLUDE']] if 'BOOST_INCLUDE' in os.environ \
                                                 else [])
LIBRARY_DIRS = [os.environ['LD_LIBRARY_PATH']] if 'LD_LIBRARY_PATH' in os.environ \
                                               else []
def find_library_dirs():
    lst = []
    try:
        os.system("whereis libboost_python | cut -d' ' -f 2 > np-include.info")
        x = open("np-include.info", "r").read().strip()
        lst.append(x[:x.index('libboost')])
    except:
        pass
    os.system("python3-config --exec-prefix > np-include.info")
    lst.append(open("np-include.info", "r").read().strip() + "/lib")
    return lst

def find_dep_lib_dirs():
    lst = []
    try:
        os.system('ldconfig -p | grep "libboost_python.*3.*" | sort -r | head -n1 | cut -d\">\" -f2 | xargs > np-include.info')
        x = open("np-include.info", "r").read().strip()
        lst.append(x[:x.index('libboost')])
    except:
        pass
    os.system("python3-config --exec-prefix > np-include.info")
    lst.append(open("np-include.info", "r").read().strip() + "/lib")
    return lst

def find_dep_lib_name(l = None):
    try:
        if l is None:
            os.system('ldconfig -p | grep "libboost_python.*3.*" | sort -r | head -n1 | cut -d\'>\' -f1 | xargs > np-include.info')
        else:
            os.system("ls " + l + "/libboost_pytho*3* | sort -r | head -n1 | cut -d'>' -f1 | xargs > np-include.info")
        x = open("np-include.info", "r").read().strip()
        return x[x.index("libboost_python"): x.index(DYNAMIC_LIB)][3:]
    except:
        return "boost_python3"

def find_numpy_lib(l = None):
    try:
        if l is None:
            os.system('ldconfig -p | grep "libboost_numpy.*3.*" | sort -r | head -n1 | cut -d\'>\' -f1 | xargs > np-include.info')
        else:
            os.system("ls " + l + "/libboost_numpy*3* | sort -r | head -n1 | cut -d'>' -f1 | xargs > np-include.info")
        x = open("np-include.info", "r").read().strip()
        return x[x.index("libboost_numpy"): x.index(DYNAMIC_LIB)][3:]
    except:
        return "boost_numpy3"

def find_boost_version():
    try:
        if 'BOOST_INCLUDE' in os.environ:
            os.system("cat " + os.environ['BOOST_INCLUDE'] + "/boost/version.hpp | grep \"#define BOOST_LIB_VERSION\" | awk '{print $3}' > np-include.info")
            return int(open("np-include.info", "r").read().strip()[1:5].replace('_', ''))
        os.system("cat $(whereis boost | awk '{print $2}')/version.hpp | grep \"#define BOOST_LIB_VERSION\" | awk '{print $3}' > np-include.info")
        return int(open("np-include.info", "r").read().strip()[1:5].replace('_', ''))
    except:
        return 165

def find_includes():
    try:
        os.system("whereis boost | awk '{print $2}' > np-include.info")
        return open("np-include.info", "r").read().strip() + '/..'
    except:
        return '/usr/include'


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
        if 'LD_LIBRARY_PATH' in os.environ:
            ALL_LIBRARIES.append(find_dep_lib_name(os.environ['LD_LIBRARY_PATH']))
        else:    
            ALL_LIBRARIES.append(find_dep_lib_name())
    else:
        copy_files(FILES, npath('source-cpp/.NEW'))
        LIBRARY_DIRS += find_library_dirs()
        if 'LD_LIBRARY_PATH' in os.environ:
            ALL_LIBRARIES.append(find_dep_lib_name(os.environ['LD_LIBRARY_PATH']))
            ALL_LIBRARIES.append(find_numpy_lib(os.environ['LD_LIBRARY_PATH']))
        else:
            ALL_LIBRARIES.append(find_dep_lib_name())
            ALL_LIBRARIES.append(find_numpy_lib())
        if platform.system() == 'Darwin':
            copy_files(FILES, npath('source-cpp/.MAC'))
    INCLUDE_DIRS.append(find_includes())

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
        version="1.2.1",
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
        packages=['pyBKT', 'pyBKT.generate', 'pyBKT.fit', 'pyBKT.util', 'pyBKT.models'],
        package_dir = { 'pyBKT': npath('source-cpp/pyBKT'),
                        'pyBKT.generate': npath('source-cpp/pyBKT/generate'),
                        'pyBKT.fit': npath('source-cpp/pyBKT/fit'),
                        'pyBKT.util': npath('source-cpp/pyBKT/util'),
                        'pyBKT.models': npath('source-cpp/pyBKT/models')},
        install_requires = ["numpy", "sklearn", "pandas"],
        ext_modules = [module1, module2, module3]
    )
except:
    setup(
        name="pyBKT",
        version="1.2.1",
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
        packages=['pyBKT', 'pyBKT.generate', 'pyBKT.fit', 'pyBKT.util', 'pyBKT.models'],
        package_dir = { 'pyBKT': npath('source-py/pyBKT'),
                        'pyBKT.generate': npath('source-py/pyBKT/generate'),
                        'pyBKT.fit': npath('source-py/pyBKT/fit'),
                        'pyBKT.util': npath('source-py/pyBKT/util'),
                        'pyBKT.models': npath('source-py/pyBKT/models')},
        install_requires = ["numpy", "sklearn", "pandas"],
    )
