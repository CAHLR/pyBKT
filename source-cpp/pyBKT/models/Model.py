#########################################
# Model.py                              #
# Model                                 #
#                                       #
# @author Anirudhan Badrinath           #
# Last edited: 07 April 2021            #
#########################################

import numpy as np
import numbers
import os
import pandas as pd
import random
import pickle
import urllib.request as urllib2
from pyBKT.generate import synthetic_data, random_model_uni
from pyBKT.fit import EM_fit, predict_onestep
from pyBKT.util import crossvalidate, data_helper, check_data, metrics

pd.options.display.float_format = '{:,.5f}'.format

class Model:
    MODELS_BKT = ['multilearn', 'multiprior', 'multipair', 'multigs']
    MODEL_ARGS = ['parallel', 'num_fits', 'seed', 'defaults'] + MODELS_BKT
    FIT_ARGS = ['skills', 'num_fits', 'defaults', 'fixed',
                            'parallel', 'forgets', 'preload'] + MODELS_BKT
    CV_ARGS = FIT_ARGS + ['folds', 'seed']
    DEFAULTS = {'num_fits': 5,
                'defaults': None,
                'parallel': True,
                'skills': '.*',
                'seed': random.randint(0, 1e8),
                'folds': 5,
                'forgets': False,
                'fixed': None,
                'model_type': [False] * len(MODELS_BKT)}
    DEFAULTS_BKT = {'order_id': 'order_id',
                    'skill_name': 'skill_name',
                    'correct': 'correct',
                    'user_id': 'user_id',
                    'multilearn': 'template_id',
                    'multiprior': 'correct',
                    'multipair': 'problem_id',
                    'multigs': 'template_id',
                    'folds': 'template_id'}
    INITIALIZABLE_PARAMS = ['prior', 'learns', 'guesses', 'slips', 'forgets']

    def __init__(self, **kwargs):
        """
        Constructs a BKT Model. Takes arguments parallel, num_fits, seed, defaults,
        and any model variant(s) that may be used. Note that all of these can be modified
        during fit/crossvalidation time.

        >>> model = Model(seed = 42)
        >>> model
        Model(parallel=True, num_fits=5, seed=42, defaults=None)

        """
        self.fit_model = None
        self.manual_param_init = False
        self._check_args(Model.MODEL_ARGS, kwargs)
        self.keep = {}
        self._update_param(['parallel', 'num_fits', 'seed', 'defaults'], kwargs, keep = True)
        self._update_param('model_type', self._update_defaults(kwargs), keep = True)

    def fit(self, data_path = None, data = None, **kwargs):
        """
        Fits a BKT model given model and data information. Takes arguments skills,
        number of initialization fits, default column names (i.e. correct, skill_name),
        parallelization, and model types. Resets model state if uninitialized.

        >>> model = Model(seed = 42)
        >>> model.fit(data_path = 'as.csv', forgets = True, skills = 'Box and Whisker')
        >>> model.evaluate(data_path = 'as.csv', metric = 'auc')
        0.6128265543747811

        """
        if not self.manual_param_init:
            self.fit_model = {}
        self.partial_fit(data_path = data_path, data = data, **kwargs)

    def partial_fit(self, data_path = None, data = None, **kwargs):
        """
        Partially fits a BKT model given model and data information. Takes arguments skills,
        number of initialization fits, default column names (i.e. correct, skill_name),
        parallelization, and model types. Behaviour is to ignore if the model is changed
        between partial fits since parameters are copied but data is reprocessed. Note
        that model type need not be specified when using partial fit after the first partial
        fit.

        >>> model = Model(seed = 42)
        >>> model.partial_fit(data_path = 'as.csv', forgets = True, skills = 'Box and Whisker')
        >>> model.partial_fit(data_path = 'as.csv', forgets = True, skills = 'Box and Whisker')
        >>> model.partial_fit(data_path = 'as.csv', forgets = True, skills = 'Box and Whisker')
        >>> model.evaluate(data_path = 'as.csv', metric = 'auc')
        0.6579800168987382

        """
        self._check_data(data_path, data)
        self._check_args(Model.FIT_ARGS, kwargs)
        self._update_param(['skills', 'num_fits', 'defaults', 'fixed',
                            'parallel', 'forgets'], kwargs)
        if self.fit_model is None or self.fit_model == {}:
            self.fit_model = {}
        if self.fit_model == {} or (self.manual_param_init and self.fit_model):
            self._update_param('model_type', self._update_defaults(kwargs))
        self.manual_param_init = True
        all_data = self._data_helper(data_path, data, self.defaults, self.skills, self.model_type)
        self._update_param(['skills'], {'skills': list(all_data.keys())})
        for skill in all_data:
            self.fit_model[skill] = self._fit(all_data[skill], skill, self.forgets, 
                                              preload = kwargs['preload'] if 'preload' in kwargs else False)
        self.manual_param_init = False

    def predict(self, data_path = None, data = None):
        """
        Predicts using the trained BKT model and test data information. Takes test data
        location or DataFrame as arguments. Returns a dictionary mapping skills to predicted
        values for those skills. Note that the predicted values are a tuple of
        (correct_predictions, state_predictions).

        >>> model = Model(seed = 42)
        >>> model.fit(data_path = 'as.csv', forgets = True, skills = 'Box and Whisker')
        >>> preds_df = model.predict(data_path = 'as.csv')
        >>> preds_df[preds_df['skill_name'] == 'Box and Whisker'][['user_id', 'correct', 'correct_predictions', 'state_predictions']]
              user_id  correct  correct_predictions  state_predictions
        0       64525        1              0.69205            0.28276
        1       64525        1              0.80226            0.10060
        2       70363        0              0.69205            0.28276
        3       70363        1              0.54989            0.51775
        4       70363        0              0.74196            0.20028
        ...       ...      ...                  ...                ...
        3952    96297        1              0.84413            0.03139
        3953    96297        1              0.84429            0.03113
        3954    96297        1              0.84432            0.03108
        3955    96298        1              0.69205            0.28276
        3956    96298        1              0.80226            0.10060

        [3957 rows x 4 columns]

        """
        self._check_data(data_path, data)
        if self.fit_model is None:
            raise ValueError("model has not been fitted yet")
        all_data, df = self._data_helper(data_path = data_path, data = data,
                             defaults = self.defaults, skills = self.skills,
                             model_type = self.model_type, gs_ref = self.fit_model,
                             resource_ref = self.fit_model,
                             return_df = True)
        # default best effort prediction of 0.5
        df['correct_predictions'] = 0.5
        df['state_predictions'] = 0.5
        for skill in all_data:
            correct_predictions, state_predictions = self._predict(self.fit_model[skill], all_data[skill])
            state_predictions = state_predictions[1]
            if all_data[skill]['multiprior_index'] is not None:
                correct_predictions = np.delete(correct_predictions, all_data[skill]['multiprior_index'])
                state_predictions = np.delete(state_predictions, all_data[skill]['multiprior_index'])
            df.loc[all_data[skill]['index'], 'correct_predictions'] = correct_predictions
            df.loc[all_data[skill]['index'], 'state_predictions'] = state_predictions
        return df

    def evaluate(self, data = None, data_path = None, metric = metrics.rmse):
        """
        Evaluates a BKT model given model and data information. Takes a metric and
        data location or DataFrame as arguments. Returns the value of the metric
        for the given trained model tested on the given data.

        >>> model = Model(seed = 42)
        >>> model.fit(data_path = 'as.csv', forgets = True, skills = 'Box and Whisker')
        >>> model.evaluate(data_path = 'as.csv', metric = 'auc')
        0.6128265543747811

        """
        self._check_data(data_path, data)
        if not isinstance(metric, (tuple, list)):
            metric = [metric]
        if self.fit_model is None:
            raise ValueError("model has not been fitted yet")
        else:
            for i in range(len(metric)):
                m = metric[i]
                if isinstance(m, str):
                    if not m in metrics.SUPPORTED_METRICS:
                        raise ValueError("metric must be one of: " + ", ".join(metrics.SUPPORTED_METRICS))
                    metric[i] = metrics.SUPPORTED_METRICS[m] 
                elif not callable(m):
                    raise ValueError("metric must either be a string, function or list/tuple of strings and functions")

        all_data = self._data_helper(data_path, data, self.defaults, self.skills, self.model_type,
                                     gs_ref = self.fit_model, resource_ref = self.fit_model)
        results = self._evaluate(all_data, metric)
        return results[0] if len(results) == 1 else results

    def crossvalidate(self, data = None, data_path = None, metric = metrics.rmse, **kwargs):
        """
        Crossvalidates (trains and evaluates) the BKT model. Takes the data, metric, and any
        arguments that would be passed to the fit function (skills, number of initialization fits, 
        default column names, parallelization, and model types) as arguments.

        >>> model = Model(seed = 42)
        >>> model.crossvalidate(data_path = 'as.csv')
                                                            mean_error
        skill
        Circle Graph                                           0.45840
        Percent Of                                             0.38005
        Finding Percents                                       0.42757
        Equivalent Fractions                                   0.45754
        Proportion                                             0.45437
        ...                                                        ...
        Solving Systems of Linear Equations                    0.38976
        Simplifying Expressions positive exponents             0.46494
        Parts of a Polyomial, Terms, Coefficient, Monom...     0.33278
        Finding Slope From Equation                            0.30684
        Recognize Quadratic Pattern                            0.00000

        [110 rows x 1 columns]

        """
        metric_names = []
        if not isinstance(metric, (tuple, list)):
            metric = [metric]
        if not isinstance(data, pd.DataFrame) and not isinstance(data_path, str):
            raise ValueError("no data specified")
        else:
            for i in range(len(metric)):
                m = metric[i]
                if isinstance(m, str):
                    if not m in metrics.SUPPORTED_METRICS:
                        raise ValueError("metric must be one of: " + ", ".join(metrics.SUPPORTED_METRICS))
                    metric[i] = metrics.SUPPORTED_METRICS[m]
                    metric_names.append(m)
                elif callable(m):
                    metric_names.append(m.__name__)
                else:
                    raise ValueError("metric must either be a string, function or list/tuple of strings and functions")

        self._check_args(Model.CV_ARGS, kwargs)
        self._update_param(['skills', 'num_fits', 'defaults', 
                            'parallel', 'forgets', 'seed', 'folds'], kwargs)
        self._update_param('model_type', self._update_defaults(kwargs))
        metric_vals = {}
        if not self.manual_param_init:
            self.fit_model = {}
        if isinstance(self.folds, str):
            self._update_defaults({'folds': self.folds})
        all_data = self._data_helper(data_path, data, self.defaults, self.skills, self.model_type, folds = isinstance(self.folds, str))
        self._update_param(['skills'], {'skills': list(all_data.keys())})
        for skill in all_data:
            metric_vals[skill] = self._crossvalidate(all_data[skill], skill, metric)
        self.manual_param_init = False
        self.fit_model = {}
        df = pd.DataFrame(metric_vals.items())
        df.columns = ['skill', 'dummy']
        df[metric_names] = pd.DataFrame(df['dummy'].tolist(), index = df.index)
        return df.set_index('skill').drop(columns = 'dummy')

    @property
    def coef_(self):
        """
        Returns the learned or preset parameters in the BKT model. Errors if model
        has not been fit or initialized. Note that the parameters are unique for
        each trained skill.

        >>> model = Model(seed = 42)
        >>> model.fit(data_path = 'as.csv', forgets = True, skills = 'Box and Whisker')
        >>> model.coef_
        {'Box and Whisker': {'learns': array([0.17969027]), 'forgets': array([0.01269486]), 'guesses': array([0.26595481]), 'slips': array([0.14831746]), 'prior': 0.8268892896231745}}

        """
        if not self.fit_model:
            raise ValueError("model has not been trained or initialized")
        return {skill: {param: self.fit_model[skill][param] for param in Model.INITIALIZABLE_PARAMS
                        if param in self.fit_model[skill]}
                for skill in self.fit_model}

    @coef_.setter
    def coef_(self, values, fixed = None):
        """
        Sets or initializes parameters in the BKT model. Values must be organized
        by skill and the BKT parameters as follows: {skill_name: {'learns': ..., 'guesses': ...}.
        Note that all parameters except the prior must be NumPy arrays.

        >>> model = Model(seed = 42)
        >>> model.coef_ = {'Box and Whisker': {'prior': 0.5}}
        >>> model.coef_
        {'Box and Whisker': {'prior': 0.5}}
        >>> model.fit(data_path = 'as.csv', forgets = True, skills = 'Box and Whisker')
        >>> model.coef_
        {'Box and Whisker': {'prior': 0.8221172842316857, 'learns': array([0.17918678]), 'guesses': array([0.27305474]), 'slips': array([0.14679578]), 'forgets': array([0.01293728])}}

        """
        self.fit_model = {}
        for skill in values:
            if skill not in self.fit_model:
                self.fit_model[skill] = {}
            if not self._check_params(values[skill]):
                raise ValueError("error in length, type or non-existent parameter")
            for param in values[skill]:
                self.fit_model[skill][param] = values[skill][param]
        self.manual_param_init = True

    def params(self):
        """ 
        Returns a DataFrame containing fitted parameters for easy
        printing.

        >>> model = Model(seed = 42)
        >>> model.fit(data_path = 'as.csv', multilearn = True, forgets = True, skills = 'Box and Whisker')
        >>> model.params()
                                          value
        skill           param   class          
        Box and Whisker prior   default 0.67443
                        learns  30799   0.16737
                                30059   0.33788
                                30060   0.28723
                                63448   0.10231
                                63447   0.07025
                                63446   0.13453
                        guesses default 0.31793
                        slips   default 0.12543
                        forgets 30799   0.00000
                                30059   0.04908
                                30060   0.01721
                                63448   0.03895
                                63447   0.00000
                                63446   0.01058

        """
        coefs = self.coef_
        formatted_coefs = []
        for skill in coefs:
            for param in coefs[skill]:
                classes = self._format_param(skill, param, coefs[skill][param])
                for class_ in classes:
                   formatted_coefs.append((skill, param, str(class_), classes[class_]))
        df = pd.DataFrame(formatted_coefs)
        df.columns = ['skill', 'param', 'class', 'value']
        return df.set_index(['skill', 'param', 'class'])

    def save(self, loc):
        """
        Saves a model to disk. Uses Python pickles.

        >>> model = Model(seed = 42)
        >>> model.fit(data_path = 'as.csv', multilearn = True, forgets = True, skills = 'Box and Whisker')
        >>> model.save('model.pkl')
        """
        with open(loc, 'wb') as handle:
            pickle.dump(self, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def load(self, loc):
        """
        Loads model given by loc into the current model.

        >>> model = Model(seed = 42)
        >>> model.load('model.pkl')
        """
        with open(loc, 'rb') as handle:
            orig_model = pickle.load(handle)
        for attr in vars(orig_model):
            setattr(self, attr, getattr(orig_model, attr))


    def fetch_dataset(self, link, loc):
        """
        Fetches dataset from an online link. Must be accessible without password
        or other authentication. Saves to the given location.

        >>> model = Model()
        >>> model.fetch_dataset('https://raw.githubusercontent.com/CAHLR/pyBKT-examples/master/data/as.csv', '.')
        """
        file_data = urllib2.urlopen(link)
        name = link.split('/')[-1]
        with open(os.path.normpath(loc + '/' + name), 'wb') as f:
            f.write(file_data.read())

    def _data_helper(self, data_path, data, defaults, skills, model_type, gs_ref = None, resource_ref = None, return_df = False, folds = False):
        """ Processes data given defaults, skills, and the model type. """
        if isinstance(data_path, str):
            data_p = data_helper.convert_data(data_path, skills, defaults = defaults, model_type = model_type, 
                                              gs_refs = gs_ref, resource_refs = resource_ref, return_df = return_df, folds = folds)
        elif isinstance(data, pd.DataFrame):
            data_p = data_helper.convert_data(data, skills, defaults = defaults, model_type = model_type,
                                                gs_refs = gs_ref, resource_refs = resource_ref, return_df = return_df, folds = folds)
        if not return_df:
            for d in data_p.values():
                check_data.check_data(d)
        else:
            for d in data_p[0].values():
                check_data.check_data(d)
        return data_p

    def _fit(self, data, skill, forgets, preload = False):
        """ Helper function for fitting data. """
        num_learns = len(data["resource_names"])
        num_gs = len(data["gs_names"])
        self._check_manual_param_init(num_learns, num_gs, skill)
        if hasattr(self, "fixed"):
            self._check_fixed(self)
        num_fit_initializations = self.num_fits
        best_likelihood = float("-inf")
        best_model = None

        for i in range(num_fit_initializations):
            fitmodel = random_model_uni.random_model_uni(num_learns, num_gs)
            optional_args = {'fixed': {}}
            if forgets:
                fitmodel["forgets"] = np.random.uniform(size = fitmodel["forgets"].shape)
            if self.model_type[Model.MODELS_BKT.index('multiprior')]:
                fitmodel["prior"] = 0
            if self.manual_param_init and skill in self.fit_model:
                for var in self.fit_model[skill]:
                    if self.fixed is not None and skill in self.fixed and var in self.fixed[skill] and \
                            isinstance(self.fixed[skill][var], bool) and self.fixed[skill][var]:
                        optional_args['fixed'][var] = self.fit_model[skill][var] 
                    elif var in fitmodel:
                        fitmodel[var] = self.fit_model[skill][var]
            if hasattr(self, "fixed") and self.fixed is not None and skill in self.fixed:
                for var in self.fixed[skill]:
                    if not isinstance(self.fixed[skill][var], bool):
                        optional_args['fixed'][var] = self.fixed[skill][var]
            if not preload:
                fitmodel, log_likelihoods = EM_fit.EM_fit(fitmodel, data, parallel = self.parallel, **optional_args)
                if log_likelihoods[-1] > best_likelihood:
                    best_likelihood = log_likelihoods[-1]
                    best_model = fitmodel
            else:
                best_model = fitmodel
        fit_model = best_model
        fit_model["learns"] = fit_model["As"][:, 1, 0]
        fit_model["forgets"] = fit_model["As"][:, 0, 1]
        fit_model["prior"] = fit_model["pi_0"][1][0]
        fit_model["resource_names"] = data["resource_names"]
        fit_model["gs_names"] = data["gs_names"]
        return fit_model
    
    def _predict(self, model, data):
        """ Helper function for predicting. """
        return predict_onestep.run(model, data)

    def _evaluate(self, all_data, metric):
        """ Helper function for evaluating. """
        per_skill = []
        true, pred = [], []
        for skill in all_data:
            correct_predictions, state_predictions = self._predict(self.fit_model[skill], all_data[skill])
            real_data = all_data[skill]['data']
            true = np.append(true, real_data.sum(axis = 0))
            pred = np.append(pred, correct_predictions)
        true = true - 1
        try:
            res = [m(true, pred) for m in metric]
        except ValueError:
            res = [m(true, pred.round(0)) for m in metric]
        return res

    def _crossvalidate(self, data, skill, metric):
        """ Helper function for crossvalidating. """
        if isinstance(self.folds, str):
            return crossvalidate.crossvalidate(self, data, skill, self.folds, metric, self.seed, True)
        else:
            return crossvalidate.crossvalidate(self, data, skill, self.folds, metric, self.seed)

    def _format_param(self, skill, param, value):
        """ Formats parameter for nice printing. """
        if isinstance(value, np.ndarray):
            ptype = 'resource_names' if (param == 'learns' or param == 'forgets') \
                                     else 'gs_names'
            names = [str(i) for i in self.fit_model[skill][ptype]]
            return dict(sorted(zip(names, value)))
        else:
            return {'default': value}

    def _check_fixed(self, fixed):
        """ Checks fixed parameter. """
        if self.fixed is None:
            return
        elif isinstance(self.fixed, bool) and self.fixed:
            self.fixed = self.fit_model
        elif isinstance(self.fixed, dict):
            return
        else:
            raise ValueError("fixed parameter incorrectly specified")


    def _update_param(self, params, args, keep = False):
        """ Updates parameters given kwargs. """
        if isinstance(args, dict):
            for param in params:
                if param not in args and (param not in self.keep or not self.keep[param]):
                    setattr(self, param, Model.DEFAULTS[param])
                elif param in args:
                    setattr(self, param, args[param])
                self.keep[param] = keep
        else:
            setattr(self, params, args)
            self.keep[params] = keep

    def _update_defaults(self, defaults):
        """ Update the default column names. """
        model_types = [False] * 4
        for d in defaults:
            if d in Model.MODELS_BKT:
                if isinstance(defaults[d], bool):
                    model_types[Model.MODELS_BKT.index(d)] = defaults[d]
                elif isinstance(defaults[d], str):
                    if self.defaults is None:
                        self.defaults = {}
                    self.defaults[d] = defaults[d]
                    model_types[Model.MODELS_BKT.index(d)] = True
                else:
                    raise ValueError("model type must either be boolean for automatic column inference" + \
                                     " or string specifying column")
            elif d in Model.DEFAULTS_BKT:
                if self.defaults is None:
                    self.defaults = {}
                self.defaults[d] = defaults[d]
        return model_types

    def _check_params(self, params):
        """ Checks if BKT parameters are valid. """
        valid = True
        for param in params:
            if param == 'prior':
                valid = valid and isinstance(params[param], numbers.Number)
            else:
                valid = valid and isinstance(params[param], np.ndarray) \
                                   and param in Model.INITIALIZABLE_PARAMS
        if 'learns' in params and 'forgets' in params:
            valid = valid and (len(params['learns']) == len(params['forgets']))
        if 'guesses' in params and 'slips' in params:
            valid = valid and (len(params['slips']) == len(params['guesses']))
        return valid

    def _check_manual_param_init(self, num_learns, num_gs, skill):
        if self.fit_model and skill in self.fit_model and 'learns' in self.fit_model[skill] \
                and len(self.fit_model[skill]['learns']) != num_learns:
            raise ValueError("invalid number of learns in initialization")
        if self.fit_model and skill in self.fit_model and 'guesses' in self.fit_model[skill] \
                and len(self.fit_model[skill]['guesses']) != num_gs:
            raise ValueError("invalid number of guess classes in initialization")
        if self.fit_model and skill in self.fit_model and 'slips' in self.fit_model[skill] \
                and len(self.fit_model[skill]['slips']) != num_gs:
            raise ValueError("invalid number of slip classes in initialization")

    def _check_args(self, expected_args, args):
        for arg in args:
            if arg not in expected_args:
                raise ValueError("provided arguments are not recognized. they must be one or more of: " + \
                        ", ".join(expected_args))

    def _check_data(self, data_path, data):
        if not isinstance(data_path, str) and not isinstance(data, pd.DataFrame):
            raise ValueError("no data specified")
        elif isinstance(data_path, str) and isinstance(data, pd.DataFrame):
            raise ValueError("cannot specify both data location and data")
        elif isinstance(data_path, str) and not os.path.exists(data_path):
            raise ValueError("data path is invalid or file not found")

    def __repr__(self):
        ret = 'Model('
        args = ['%s=%s' % (arg, str(getattr(self, arg))) for arg in Model.MODEL_ARGS if hasattr(self, arg)]
        ret += ', '.join(args) + ')'
        return ret
