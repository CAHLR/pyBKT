# pyBKT

Python implementation of the Bayesian Knowledge Tracing algorithm and variants, estimating student cognitive mastery from problem solving sequences.
```
    pip install pyBKT
```
[Quick-start example in Colab notebook](https://colab.research.google.com/drive/1TKJkKYPAIub5jJSpAe04HJmP08EFYtMV?usp=sharing "pyBKT quick start in Colab") 

Based on the work of Zachary A. Pardos (zp@berkeley.edu) and Matthew J. Johnson (mattjj@csail.mit.edu) @ https://github.com/CAHLR/xBKT. Python boost adaptation by Cristian Garay (c.garay@berkeley.edu). All-platform python adaptation and optimizations by Anirudhan Badrinath (abadrinath@berkeley.edu). For formulas and technical implementation details, please refer to section 4.3 of Xu, Johnson, & Pardos (2015) ICML workshop [paper](http://ml4ed.cc/attachments/XuY.pdf). 

## Requirements
Python >= 3.5

Supported OS: All platforms! (Yes, Windows too)

Libboost >= 1.58 (optional - will enable fast inference if installed)
## Supported model variants
pyBKT can be used to define and fit many BKT variants, including these from the literature: 

* Individual student priors, learn rate, guess, and slip [1,2]
* Individual item guess and slip [3,4,5]
* Individual item or resource learn rate [4,5]

1. Pardos, Z. A., Heffernan, N. T. (2010) Modeling Individualization in a Bayesian Networks Implementation of Knowledge Tracing. In P. De Bra, A. Kobsa, D. Chin (Eds.) *Proceedings of the 18th International Conference on User Modeling, Adaptation and Personalization* (UMAP). Big Island of Hawaii. Pages. Springer. Pages 255-266. [[doi]](https://doi.org/10.1007/978-3-642-13470-8_24
)

1. Pardos, Z. A., Heffernan, N. T. (2010) Using HMMs and bagged decision trees to leverage rich features of user and skill from an intelligent tutoring system dataset. In J. Stamper & A. Niculescu-Mizil (Eds.) *Proceedings of the KDD Cup Workshop at the 16th ACM Conference on Knowledge Discovery and Data Mining* (SIGKDD). Washington, D.C. ACM. Pages 24-35. [[kdd cup]](https://pslcdatashop.web.cmu.edu/KDDCup/workshop/papers/pardos_heffernan_KDD_Cup_2010_article.pdf)

1. Pardos, Z. & Heffernan, N. (2011) KT-IDEM: Introducing Item Difficulty to the Knowledge Tracing Model. In Konstant et al. (eds.) *Proceedings of the 20th International Conference on User Modeling, Adaptation and Personalization* (UMAP). Girona, Spain. Springer. Pages 243-254. [[doi]](https://doi.org/10.1007/978-3-642-22362-4_21)

1. Pardos, Z. A., Bergner, Y., Seaton, D., Pritchard, D.E. (2013) Adapting Bayesian Knowledge Tracing to a Massive Open Online College Course in edX. In S.K. D’Mello, R.A. Calvo, & A. Olney (Eds.) *Proceedings of the 6th International Conference on Educational Data Mining* (EDM). Memphis, TN. Pages 137-144. [[edm]](http://educationaldatamining.org/EDM2013/proceedings/paper_20.pdf)

1. Xu, Y., Johnson, M. J., Pardos, Z. A. (2015) Scaling cognitive modeling to massive open environments. In *Proceedings of the Workshop on Machine Learning for Education at the 32nd International Conference on Machine Learning* (ICML). Lille, France. [[icml ml4ed]](http://ml4ed.cc/attachments/XuY.pdf)



# Installation and setup
This is intended as a quick overview of steps to install and setup and to run pyBKT locally.

## Installing Boost-Python ##

If you have Lib Boost already installed, pip will install pyBKT with fast c++ inferencing. Boost is already installed on Ubuntu distributions. If it is not installed on your machine, type `sudo apt install libboost-all-dev`. Use whichever package manager is appropriately suited to your distribution. Without Boot, pip will install pyBKT without c++ speed optimizations.

## Installing ##

Once `libboost` is installed (check by doing a quick `ldconfig -p | grep libboost_python`, which should yield an output), you can simply run:

```
    pip install pyBKT
``` 
Alternatively, if `pip` poses some problems, you can clone the repository as such and then run the `setup.py` script manually.

```
    git clone https://github.com/CAHLR/pyBKT.git
    cd pyBKT
    python3 setup.py install
```

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
