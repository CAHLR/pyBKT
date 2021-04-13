# pyBKT

Python implementation of the Bayesian Knowledge Tracing algorithm and variants, estimating student cognitive mastery from problem solving sequences.

```
    pip install pyBKT
```
[pyBKT Quick Start Tutorial](https://colab.research.google.com/drive/13abu919edUXbvPV3qeGPpvwnFBExU7Vd "pyBKT quick start in Colab")

[LAK pyBKT Tutorial in Google Colab Notebook](https://colab.research.google.com/drive/1Kg6AvXKdSZXoqzSZ5BRHuewyHRMvrZs1 "pyBKT quick start in Colab") 

Based on the work of Zachary A. Pardos (zp@berkeley.edu) and Matthew J. Johnson (mattjj@csail.mit.edu) @ https://github.com/CAHLR/xBKT. Python boost adaptation by Cristian Garay (c.garay@berkeley.edu). All-platform python adaptation and optimizations by Anirudhan Badrinath (abadrinath@berkeley.edu). For formulas and technical implementation details, please refer to section 4.3 of Xu, Johnson, & Pardos (2015) ICML workshop [paper](http://ml4ed.cc/attachments/XuY.pdf). 

pyBKT examples can be found in [pyBKT-examples](https://github.com/CAHLR/pyBKT-examples/ "pyBKT examples") repo.

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

We offer both a pure Python port and a Python/C++ extension version of pyBKT for the sake of accessibility and ease of use on any platform. Note that pip, by default, will install the C++/Python version unless the required libraries are not found or there is an error during installation. In the case of such issues, it will revert to the pure Python implementation.

The former pure Python versions does not fit models or scale as quickly or efficiently as the latter (due to nested for loops needed for DP). Here are a few speed comparisons - both on the same machine - that may be useful in deciding which version is more appropriate given the usage (e.g. model fitting is far more demanding than prediction). Note that the C++/Python extensions require the Boost libraries as specified below: most Linux machines should come preinstalled with them.

|                 Test Description                | pyBKT (Python) | pyBKT (Cython) |
|:-----------------------------------------------:|:--------------:|---------------:|
| synthetic data, model fit (500 students)        |     ~1m55s     |      ~1.5s     |
| synthetic data, model fit (5000 students)       |     ~1h30m     |      ~45s      |
| cross validated cognitive tutor data            |     ~4m10s     |       ~3s      |
| synthetic data, predict onestep (500 students)  |       ~2s      |      ~0.8s     |
| synthetic data, predict onestep (5000 students) |     ~2m15s     |      ~35s      |

## Installing Dependencies for Fast C++ Inferencing (Optional - for OS X and Linux before Xenial) ##

### Linux

If you have Boost already installed, pip will install pyBKT with fast C++ inferencing. Boost is already installed on most recent Ubuntu distributions. If it is not installed on your machine, type `sudo apt install libboost-all-dev` if using Debian based distributions. Otherwise, whichever package manager is appropriately suited to your distribution (`dnf`, `pacman`, etc.). Without Boost, pip will install pyBKT without C++ speed optimizations.

You can check if libboost has been installed properly with `ldconfig -p | grep libboost_python`, which should yield an output on Linux machines. Note that the version on the dynamic library should match the Python installation version.

In case this is a hassle, we provide a Conda environment that works very easily. Simply execute inside your base conda environment:

``` source setup_conda.sh ```

You may need to run the above as root.

### Mac

The latest version of Python is necessary for OS X. If homebrew is installed, run the following commands to download the necessary dependencies:
```
    brew install boost
    brew install boost-python3
    brew install libomp
```

Note that if you see an error about a symbol not being found when you import pyBKT.models.Model, you likely have a mismatched Boost and Python version. Check if libboost_pythonXX.dylib and your Python version X.X are the same (i.e. libboost_python39 and Python 3.9).

In case this is a hassle, we provide a Conda environment that works very easily. Simply execute inside your base conda environment:

``` source setup_conda.sh ```

You may need to run the above as root.

## Installing pyBKT ##

You can simply run:
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

The following serves as a mini-tutorial for how to get started with pyBKT. There is more information available at the Colab notebook listed at the top of the README.

## Input and Output Data ##

The accepted input formats are Pandas DataFrames and data files of type csv (comma separated) or tsv (tab separated). pyBKT will automatically infer which delimiter to use in the case that it is passed a data file. Since column names mapping meaning to each field in the data (i.e. skill name, correct/incorrect) vary per data source, you may need to specify a mapping from your data file's column names to pyBKT's expected column names. In many cases with Cognitive Tutor and Assistments datasets, pyBKT will be able to automatically infer column name mappings, but in the case that it is unable to, it will raise an exception. Note that the correctness is given by -1 (no response), 0 (incorrect), or 1 (correct).

## Creating and Training Models ##

The process of creating and training models in pyBKT resemble that of SciKit Learn. pyBKT provides easy methods of fetching online datasets and to fit on a combination or all skills available in any particular dataset.

```python
from pyBKT.models import Model

# Initialize the model with an optional seed
model = Model(seed = 42, num_fits = 1)

# Fetch Assistments and CognitiveTutor data (optional - if you have your own dataset, that's fine too!)
model.fetch_dataset('https://raw.githubusercontent.com/CAHLR/pyBKT-examples/master/data/as.csv', '.')
model.fetch_dataset('https://raw.githubusercontent.com/CAHLR/pyBKT-examples/master/data/ct.csv', '.')

# Train a simple BKT model on all skills in the CT dataset
model.fit(data_path = 'ct.csv')

# Train a simple BKT model on one skill in the CT dataset
# Note that calling fit deletes any previous trained BKT model!
model.fit(data_path = 'ct.csv', skills = "Plot imperfect radical")

# Train a simple BKT model on multiple skills in the CT dataset
model.fit(data_path = 'ct.csv', skills = ["Plot imperfect radical",
                                          "Plot pi"])

# Train a multiguess and slip BKT model on multiple skills in the
# CT dataset. Note: if you are not using CognitiveTutor or Assistments
# data, you may need to provide a column mapping for the guess/slip
# classes to use (i.e. if the column name is gsclasses, you would
# specify multigs = 'gsclasses' or specify a defaults dictionary
# defaults = {'multigs': 'gsclasses'}).
model.fit(data_path = 'ct.csv', skills = ["Plot imperfect radical",
                                          "Plot pi"],
                                multigs = True)

# We can combine multiple model variants.
model.fit(data_path = 'ct.csv', skills = ["Plot imperfect radical",
                                          "Plot pi"],
                                multigs = True, forgets = True,
                                multilearn = True)

# We can use a different column to specify the different learn and 
# forget classes. In this case, we use student ID.
model.fit(data_path = 'ct.csv', skills = ["Plot imperfect radical",
                                          "Plot pi"],
                                multigs = True, forgets = True,
                                multilearn = 'Anon Student Id')

# View the trained parameters!
print(model.params())
```

Note that if we train on a dataset that has unfamiliar columns to pyBKT, you will be required to specify a mapping of column names in that dataset to expected pyBKT columns. This is referred to as the model defaults (i.e. it specifies the default column names to lookup in the dataset). An example usage is provided below for an unknown dataset which has column names "row", "skill\_t", "answer", and "gs\_classes".
``` python
# Load unfamiliar dataset.
df = pd.read_csv('mystery.csv')

# For other non-Assistments/CogTutor style datasets, we will need to specify the
# columns corresponding to each required column (i.e. the user ID, correct/incorrect).
# For that, we use a defaults dictionary.
# In this case, the order ID that pyBKT expects is specified by the column row in the
# dataset, the skill_name is specified by a column skill_t and the correctness is specified
# by the answer column in the dataset.
defaults = {'order_id': 'row', 'skill_name': 'skill_t', 'correct': 'answer'}

# This defaults dictionary contains columns specifying what columns correspond
# to the desired guess/slip classes, etc. In this case, our desired column for
# the guess/slip classes is a column named gs_classes.
defaults['multigs'] = 'gs_classes'

# Fit using the defaults (column mappings) specified in the dictionary.
model.fit(data = df, defaults = defaults)

# Predict/evaluate/etc.
training_acc = model.evaluate(data = df, metric = 'accuracy')
```

## Model Prediction and Evaluation ##

Prediction and evaluation behave similarly to SciKit-Learn. pyBKT offers a variety of features for prediction and evaluation.

```python
from pyBKT.models import Model

# Initialize the model with an optional seed
model = Model(seed = 42, num_fits = 1)

# Load the Cognitive Tutor data (not necessary, but shown
# for the purposes of the tutorial that pyBKT accepts
# DataFrames as well as file locations!).
ct_df = pd.read_csv('ct.csv', encoding = 'latin')

# Train a simple BKT model on all skills in the CT dataset
model.fit(data_path = 'ct.csv')

# Predict on all skills on the training data.
# This returns a Pandas DataFrame.
preds_df = model.predict(data_path = 'ct.csv')

# Evaluate the RMSE of the model on the training data.
# Note that the default evaluate metric is RMSE.
training_rmse = model.evaluate(data = ct_df)

# Evaluate the AUC of the model on the training data. The supported
# metrics are AUC, RMSE and accuracy (they should be lowercased in
# the argument!).
training_auc = model.evaluate(data_path = 'ct.csv', metric = 'auc')

# We can define a custom metric as well.
def mae(true_vals, pred_vals):
  """ Calculates the mean absolute error. """
  return np.mean(np.abs(true_vals - pred_vals))

training_mae = model.evaluate(data_path = 'ct.csv', metric = mae)
```

## Crossvalidation ##

Crossvalidation is offered as a blackbox function similar to a combination of fit and evaluate that accepts a particular number of folds, a seed, and a metric (either one of the 3 provided that are 'rmse', 'auc' or 'accuracy' or a custom Python function taking 2 arguments). Similar arguments for the model types, data path/data, and skill names are accepted as with the fit function.

``` python
from pyBKT.models import Model

# Initialize the model with an optional seed
model = Model(seed = 42, num_fits = 1)

# Crossvalidate with 5 folds on all skills in the CT dataset.
crossvalidated_errors = model.crossvalidate(data_path = 'ct.csv', folds = 5)

# Crossvalidate on a particular set of skills with a given 
# seed, folds and metric.
def mae(true_vals, pred_vals):
  """ Calculates the mean absolute error. """
  return np.mean(np.abs(true_vals - pred_vals))

# Note that the skills argument accepts a REGEX pattern. In this case, this matches and 
# crossvalidates on all skills containing the word fraction.
crossvalidated_mae_errs = model.crossvalidate(data_path = 'ct.csv', skills = ".*fraction.*",
                                              folds = 10, metric = mae)

# Crossvalidate using multiple model variants.
crossvalidated_multigsf = model.crossvalidate(data_path = 'ct.csv', multigs = True, forgets = True)
```

## Extended Features ##

Extended features include model parameter initialization by setting model.coef_, providing a configuration dictionary, setting model default columns, and more. For more information about these features, take a look at the Colab notebook provided at the top of the README.

# Internal Data Format #

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

## TODOs ##
* Support for parameter tieing and fixing 
* Boostless Cython
