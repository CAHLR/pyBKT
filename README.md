# pyBKT

Python implementation of the Bayesian Knowledge Tracing algorithm, modeling student cognitive mastery from problem solving sequences.

Based on the work of Zachary A. Pardos (zp@berkeley.edu) and Matthew J. Johnson (mattjj@csail.mit.edu) @ https://github.com/CAHLR/xBKT. Python adaptation by Cristian Garay (c.garay@berkeley.edu). 

Python notebook quick start example: [Google Colab notebook](https://colab.research.google.com/drive/13iqM8mtYwtOKw3Ja4RS51ZTa4NnEQ26s "pyBKT quick start in Colab")

This implimentation can be used to define many BKT variants, including these from the literature: 

Pardos, Z. A., Heffernan, N. T. (2010) Modeling Individualization in a Bayesian Networks Implementation of Knowledge Tracing. In P. De Bra, A. Kobsa, D. Chin (eds.) Proceedings of the 18th International Conference on User Modeling, Adaptation and Personalization (UMAP). Big Island of Hawaii. Pages. Springer. Pages 255-266.

Pardos, Z. & Heffernan, N. (2011) KT-IDEM: Introducing Item Difficulty to the Knowledge Tracing Model. In Konstant et al. (eds.) Proceedings of the 20th International Conference on User Modeling, Adaptation and Personalization (UMAP). Girona, Spain. Springer. Pages 243-254.

Pardos, Z.A., Heffernan, N.T. (2012) Tutor Modeling vs. Student Modeling. In Proceedings of the 25th annual Florida Artificial Intelligence Research Society Conference. Marco Island, FL. AAAI. Pages 420-425.

This is intended as a quick overview of steps to install and setup and to run pyBKT locally.

# Instalation and setup

## Cloning the repository ##

```
git clone https://github.com/CAHLR/pyBKT.git
```

## Installing Eigen ##

Get Eigen from http://eigen.tuxfamily.org/index.php?title=Main_Page and unzip
it into the pyBKT directory:

    cd pyBKT
    wget --no-check-certificate http://bitbucket.org/eigen/eigen/get/3.1.3.tar.gz
    tar -xzvf 3.1.3.tar.gz
    mv ./eigen-eigen-2249f9c22fe8 ./Eigen
    rm 3.1.3.tar.gz

## Installing Boost-Python ##

On Unix machines, install all the Boost Libraries using `sudo apt install libboost-all-dev`

On OS X, the easiest way to install Boost-Python is through Homebrew:

```
For Python 3 (when is not the default Python version in the machine)
- brew uninstall boost-python (if already installed)
- brew install boost-python --with-python3 --without-python

For Python 2 or when Python 3 is the default Python version.
- brew install boost-python
```

NOTE: pyBKT is currently not compatible with libboost version >= 1.65.

## Compiling ##

Before running `make`, check `Makefile` in the pyBKT folder. Be sure that the paths for all the libraries are correct (Boos-Python, Eigen, Numpy, OMP). The default python version in the Makefile is 3.6. If you have a different version, you will need to change NUMPY_INCLUDE_PATH  and BOOST_PYTHON_LIBS, accordingly. 

Run `make` in the root directory of the pyBKT project folder. If this step runs successfully, you should see one _.o_ and one _.so_ file generated for each of the _.cpp_ files and the simulated data generation and training example should complete successfully, printing ground truth and learned BKT parameter values.

## Potential Errors When Running Makefile on OS X ##

You may see the following error while running `make`
```
    make: g++-4.9: No such file or directory
```

Try `gcc --version` in your terminal. If a version exists, you already have gcc installed. This error may be due to an incorrect version of gcc being called. In order to change the gcc version in `Makefile`, update the `CXX` variable. For example, you may need to change `CXX=g++-4.9` to `CXX=g++-5`, depending on the version you set up. 

If a version does not exist, you  may need to download gcc49. This can be downloaded with [brew](http://brew.sh/). 

These steps would allow you to set up gcc49. Run the following commands
```
    brew install --enable-cxx gcc49
    brew install mpfr
    brew install gmp
    brew install libmpc
```

The Makefile uses the python-config tool, if need to install it run: `sudo apt install python-dev`

# Preparing Data and Running Model #
## Input and Output Data ##
_pyBKT_ models student mastery of a skills as they progress through series of learning resources and checks for understanding. Mastery is modelled as a latent variable has two states - "knowing" and "not knowing". At each checkpoint, students may be given a learning resource (i.e. watch a video) and/or question(s) to check for understanding. The model finds the probability of learning, forgetting, slipping and guessing that maximizes the likelihood of observed student responses to questions. 

To run the pyBKT model, define the following variables:
* `num_subparts`: The number of unique questions used to check understanding. Each subpart has a unique set of emission probabilities.
* `num_resources`: The number of unique learning resources available to students.
* `num_fit_initialization`: The number of iterations in the EM step.


Next, create an input object `Data`, containing the following attributes: 
* `data`: a matrix containing sequential checkpoints for all students, with their responses. Each row represents a different subpart, and each column a checkpoint for a student. There are three potential values: {0 = no response or no question asked, 1 = wrong response, 2 = correct response}. If at a checkpoint, a resource was given but no question asked, the associated column would have `0` values in all rows. For example, to set up data containing 5 subparts given to two students over 2-3 checkpoints, the matrix would look as follows:

        | 0  0  0  0  2 |
        | 0  1  0  0  0 |
        | 0  0  0  0  0 |
        | 0  0  0  0  0 |
        | 0  0  2  0  0 |   

  In the above example, the first student starts out with just a learning resource, and no checks for understanding. In subsequent checkpoints, this student also responds to subpart 2 and 5, and gets the first wrong and the second correct.   

* `starts`: defines each student's starting column on the `data` matrix. For the above matrix, `starts` would be defined as: 

        | 1  4 |

* `lengths`: defines the number of check point for each student. For the above matrix, `lengths` would be defined as: 

        | 3  2 |

* `resources`: defines the sequential id of the resources at each checkpoint. Each position in the vector corresponds to the column in the `data` matrix. For the above matrix, the learning `resources` at each checkpoint would be structured as: 

        | 1  2  1  1  3 |

* `stateseqs`: this attribute is the true knowledge state for above data and should be left undefined before running the `pyBKT` model. 


The output of the model can will be stored in a `fitmodel` object, containing the following probabilities as attributes: 
* `As`: the transition probability between the "knowing" and "not knowing" state. Includes both the `learns` and `forgets` probabilities, and their inverse. `As` creates a separate transition probability for each resource.
* `learns`: the probability of transitioning to the "knowing" state given "not known".
* `forgets`: the probability of transitioning to the "not knowing" state given "known".
* `prior`: the prior probability of "knowing".

The `fitmodel` also includes the following emission probabilities:
* `guesses`: the probability of guessing correctly, given "not knowing" state.
* `slips`: the probability of picking incorrect answer, given "knowing" state.


## Running pyBKT ##
You can add the folder path to the PYTHONPATH env variable in order to run the model from anywhere in your system. In Unix-based systems edit your _.bash_rc_ or _.bash_profile_ file and add:

```
export PYTHONPATH="${PYTHONPATH}:/path_to_folder_containing_pyBKT_folder"
```

To start the EM algorithm, initiate a randomly generated `fitmodel`, with two potential options:

1. `generate.random_model_uni`: generates a model from uniform distribution and sets the `forgets` probability to 0.

2. `generate.random_model`: generates a model from dirichlet distribution and allows the `forgets` probability to vary. 

For data observed during a short period of learning activity with a low probability of forgetting, the uniform model is recommended. The following example will initiate fitmodel using the uniform distribution: 

         fitmodel = random_model.random_model_uni(num_resources, num_subparts)

Once the `fitmodel` is generated, the following function can be used to generate an updated `fitmodel` and `log_likelihoods`:

        (fitmodel, log_likelihoods) = EM_fit.EM_fit(fitmodel, data)

## Example ##
[TODO: Update Example Model]

See the file `test/hand_specified_model.py` for a fairly complete example,
which you can run with `python test/hand_specified_model.py`.

Here's a simplified version:

```python
import sys
sys.path.append('../') #path containing pyBKT
import numpy as np
from pyBKT.generate import synthetic_data, random_model_uni
from pyBKT.fit import EM_fit
from copy import deepcopy

#parameters classes
num_gs = 1 #number of guess/slip classes
num_learns = 1 #number of learning rates

num_fit_initializations = 20

#true params used for synthetic data generation
p_T = 0.30
p_F = 0.00
p_G = 0.10
p_S = 0.03
p_L0 = 0.10

#generate synthetic model and data.
truemodel = {}

truemodel["As"] =  np.zeros((num_learns,2,2), dtype=np.float_)
for i in range(num_learns):
    truemodel["As"][i] = np.transpose([[1-p_T, p_T], [p_F, 1-p_F]])

truemodel["learns"] = truemodel["As"][:,1, 0,]
truemodel["forgets"] = truemodel["As"][:,0, 1]

truemodel["pi_0"] = np.array([[1-p_L0], [p_L0]])
truemodel["prior"] = truemodel["pi_0"][1][0]

truemodel["guesses"] = np.full(num_gs, p_G, dtype=np.float_)
truemodel["slips"] = np.full(num_gs, p_S, dtype=np.float_)
#can optionally set learn class sequence - set randomly by synthetic_data if not included
#truemodel["resources"] = np.random.randint(1, high = num_resources, size = sum(observation_sequence_lengths))

#data!
print("generating data...")
observation_sequence_lengths = np.full(500, 100, dtype=np.int) #specifies 500 students with 100 observations for synthetic data
data = synthetic_data.synthetic_data(truemodel, observation_sequence_lengths)

#fit models, starting with random initializations
print('fitting! each dot is a new EM initialization')

num_fit_initializations = 5
best_likelihood = float("-inf")

for i in range(num_fit_initializations):
	fitmodel = random_model_uni.random_model_uni(num_learns, num_gs) # include this line to randomly set initial param values
	(fitmodel, log_likelihoods) = EM_fit.EM_fit(fitmodel, data)
	if(log_likelihoods[-1] > best_likelihood):
		best_likelihood = log_likelihoods[-1]
		best_model = fitmodel

# compare the fit model to the true model

print('')
print('\ttruth\tlearned')
print('prior\t%.4f\t%.4f' % (truemodel['prior'], best_model["pi_0"][1][0]))
for r in range(num_learns):
    print('learn%d\t%.4f\t%.4f' % (r+1, truemodel['As'][r, 1, 0].squeeze(), best_model['As'][r, 1, 0].squeeze()))
for r in range(num_learns):
    print('forget%d\t%.4f\t%.4f' % (r+1, truemodel['As'][r, 0, 1].squeeze(), best_model['As'][r, 0, 1].squeeze()))

for s in range(num_gs):
    print('guess%d\t%.4f\t%.4f' % (s+1, truemodel['guesses'][s], best_model['guesses'][s]))
for s in range(num_gs):
    print('slip%d\t%.4f\t%.4f' % (s+1, truemodel['slips'][s], best_model['slips'][s]))

```
