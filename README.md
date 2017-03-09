# pyBKT

Python implementation of the Bayesian Knowledge Tracing algorithm to model learner's mastery of the knowledge being tutored.

Based on the work of Zachary A. Pardos (zp@berkeley.edu) and Matthew J. Johnson (mattjj@csail.mit.edu) Computational Approaches to Human Learning Research (CAHL) Lab @ UC Berkeley https://github.com/CAHLR/xBKT

This is intended as a quick overview of steps to install and setup and to run xBKT locally.

# Instalation and setup

## Cloning the repository ##

```
git clone git@github.com/CAHLR/xBKT.git
```

## Installing Eigen ##

Get Eigen from http://eigen.tuxfamily.org/index.php?title=Main_Page and unzip
it somewhere (anywhere will work, but it affects the mex command below). On a
\*nix machine, these commands should put Eigen in /usr/local/include:


    cd /usr/local/include
    wget --no-check-certificate http://bitbucket.org/eigen/eigen/get/3.1.3.tar.gz
    tar -xzvf 3.1.3.tar.gz
    ln -s eigen-eigen-2249f9c22fe8/Eigen ./Eigen
    rm 3.1.3.tar.gz

Similarly, if working in OS X, you can download the latest stable version of Eigen 
from the site above. This program has run successfully with `Eigen 3.2.5`.
First move the file to /usr/local/include, then unzip and create simplified link to Eigen. 
These commands can be used below:


    mv <path to file>/3.1.3.tar.gz /usr/local/include/3.1.3.tar.gz
    tar -xvf 3.1.3.tar.gz
    ln -s <name of unzipped file>/Eigen ./Eigen
    rm 3.1.3.tar.gz

---

Don't have python-config install python-dev (Linux): sudo apt install python-dev

Windows: Install gcc 4.9? http://preshing.com/20141108/how-to-install-the-latest-gcc-on-windows/
https://sourceforge.net/projects/mingw/

Install boost-python for python 3 through brew:
- brew uninstall boost-python (if already installed)
- brew install boost-python --with-python3 --without-python
