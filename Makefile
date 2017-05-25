#for python3

#c++ compiler
CXX = g++-4.9

#-------------------

PYTHON_INCLUDE_PATH = $(shell python3-config --includes)

O2_LIB = -O2

#where eigen was installed.
EIGEN_INCLUDE_PATH = -I/home/cgaray

#boost-python include path.
#need to get this more programmaticaly.
BOOST_PYTHON_INCLUDE_PATH = -I/usr/include

#numpy include path
#this should change for anaconda?
#need to get this more programmaticaly.
#NUMPY_INCLUDE_PATH = -I/Library/Frameworks/Python.framework/Versions/2.7/lib/python2.7/site-packages/numpy/core/include
#NUMPY_INCLUDE_PATH = -I/Users/cgaray/anaconda/lib/python3.5/site-packages/numpy/core/include/
NUMPY_INCLUDE_PATH = -I/usr/local/lib/python3.4/dist-packages/numpy/core/include/

#omp include path (?)
OMP_INCLUDE_PATH = -I/usr/lib/gcc/x86_64-linux-gnu/4.6/include/

ALL_OPTS = $(PYTHON_INCLUDE_PATH) $(O2_LIB) $(EIGEN_INCLUDE_PATH) \
$(BOOST_PYTHON_INCLUDE_PATH) $(NUMPY_INCLUDE_PATH) $(OMP_INCLUDE_PATH)

#-------------------

#change python-config for python3-config on python3 (works with Anaconda).
#or complete the paths and libraries manually.

#where the dylib files are located.
PYTHON_LIB_PATH = $(shell python3-config --exec-prefix)/lib

#basic python libs
PYTHON_LIBS = $(shell python3-config --libs)

#boost-python lib path.
#need to get this more programmaticaly.
BOOST_PYTHON_LIB_PATH = -L/usr/local/lib

#boost-python libs
BOOST_PYTHON_LIBS = -lboost_python

#openmp libs
OPENMP_LIBS = -fopenmp

ALL_LIBS = -L$(PYTHON_LIB_PATH) $(PYTHON_LIBS) $(BOOST_PYTHON_LIB_PATH) \
$(BOOST_PYTHON_LIBS) $(OPENMP_LIBS)

#-------------------

default: generate/synthetic_data_helper.so fit/E_step.so
	@python test/hand_specified_model3.py

generate/synthetic_data_helper.so: generate/synthetic_data_helper.o
	$(CXX) $< $(ALL_LIBS) -Wl,-rpath,$(PYTHON_LIB_PATH) -shared -o $@

generate/synthetic_data_helper.o: generate/synthetic_data_helper.cpp Makefile
	$(CXX) $< $(ALL_OPTS) -c -fPIC -o $@

fit/E_step.so: fit/E_step.o
	$(CXX) $< $(ALL_LIBS) -Wl,-rpath,$(PYTHON_LIB_PATH) -shared -o $@

fit/E_step.o: fit/E_step.cpp Makefile
	$(CXX) $< $(ALL_OPTS) -c -fPIC -o $@

clean:
	rm -rf generate/*.so generate/*.o fit/*.so fit/*.o

.PHONY: default clean
