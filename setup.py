#########################################
# setup.py                              #
# Setup for PyBKT                       #
#                                       #
# @author Anirudhan Badrinath           #
# Last edited: 01 April 2021            #
#########################################

import os
from os.path import normpath as npath
import sys
from sysconfig import get_paths
from setuptools import setup, Extension
import platform
from distutils.command.build_ext import build_ext

sys.tracebacklimit = 0

class CustomBuildExtCommand(build_ext):
    """build_ext command for use when numpy headers are needed."""
    """ https://stackoverflow.com/questions/2379898/make-distutils-look-for-numpy-header-files-in-the-correct-place """
    def run(self):
        import numpy
        self.include_dirs.append(numpy.get_include())
        build_ext.run(self)

FILES = {'synthetic_data_helper.cpp': 'source-cpp/pyBKT/generate/',
         'predict_onestep_states.cpp': 'source-cpp/pyBKT/fit/', 
         'E_step.cpp': 'source-cpp/pyBKT/fit/'}

if platform.system() == 'Darwin':
    ALL_COMPILE_ARGS = ['-c', '-fPIC', '-w', '-O3', '-stdlib=libc++', '-Xpreprocessor', '-fopenmp']
    ALL_LINK_ARGS = ['-stdlib=libc++']
    ALL_LIBRARIES = ['pthread', 'dl', 'util', 'm', 'omp']
else:
    ALL_COMPILE_ARGS = ['-c', '-fPIC', '-w', '-fopenmp', '-O2']
    ALL_LINK_ARGS = ['-fopenmp']
    ALL_LIBRARIES = ['pthread', 'dl', 'util', 'm']
INCLUDE_DIRS = sys.path + ['source-cpp/pyBKT/Eigen/', get_paths()['include']]
LIBRARY_DIRS = [os.environ['LD_LIBRARY_PATH']] if 'LD_LIBRARY_PATH' in os.environ \
                                               else []
def clean():
    global LIBRARY_DIRS, ALL_LIBRARIES
    LIBRARY_DIRS = [i for i in LIBRARY_DIRS if i != ""]
    ALL_LIBRARIES = [i for i in ALL_LIBRARIES if i != ""]

with open('README.md', encoding='utf-8') as f:
    long_description = f.read()

LIBRARY_DIRS += [sys.exec_prefix + '/lib']
clean()
try:
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
        version="1.4",
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
        install_requires = ["numpy", "sklearn", "pandas", "requests"],
        setup_requires = ["numpy"],
        cmdclass = {'build_ext': CustomBuildExtCommand},
        ext_modules = [module1, module2, module3]
    )
except:
# LEGACY PURE PYTHON VERSION:
    setup(
        name="pyBKT",
        version="1.4",
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
        install_requires = ["numpy", "sklearn", "pandas", "requests"],
    )
