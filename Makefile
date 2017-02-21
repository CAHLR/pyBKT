CC = g++-4.9
PYLIBPATH = $(shell python-config --exec-prefix)/lib
LIB = -L$(PYLIBPATH) $(shell python-config --libs) -L/usr/local/Cellar/boost-python/1.60.0/lib -lboost_python -fopenmp
OPTS = $(shell python-config --include) -O2 -I/usr/local/include -I/usr/local/Cellar/boost/1.60.0_1/include -I/Library/Frameworks/Python.framework/Versions/2.7/lib/python2.7/site-packages/numpy/core/include -I/usr/local/Cellar/gcc49/4.9.3/lib/gcc/4.9/gcc/x86_64-apple-darwin15.6.0/4.9.3/include

default: generate/synthetic_data_helper.so fit/E_step.so
	@python2 test/hand_specified_model3.py

generate/synthetic_data_helper.so: generate/synthetic_data_helper.o
	$(CC) $(LIB) -Wl,-rpath,$(PYLIBPATH) -shared $< -o $@

generate/synthetic_data_helper.o: generate/synthetic_data_helper.cpp Makefile
	$(CC) $(OPTS) -c $< -o $@

fit/E_step.so: fit/E_step.o
	$(CC) $(LIB) -Wl,-rpath,$(PYLIBPATH) -shared $< -o $@

fit/E_step.o: fit/E_step.cpp Makefile
	$(CC) $(OPTS) -c $< -o $@

clean:
	rm -rf generate/*.so generate/*.o fit/*.so fit/*.o

.PHONY: default clean