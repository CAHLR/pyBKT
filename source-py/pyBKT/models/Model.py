import numpy as np
import numbers
import os
import pandas as pd
import random
import urllib.request as urllib2
from pyBKT.generate import synthetic_data, random_model_uni
from pyBKT.fit import EM_fit, predict_onestep
from pyBKT.util import crossvalidate, data_helper, check_data, metrics

class Model:
    MODELS_BKT = ['multilearn', 'multiprior', 'multipair', 'multigs']
    MODEL_ARGS = ['parallel', 'num_fits', 'seed', 'defaults'] + MODELS_BKT
    FIT_ARGS = ['skills', 'num_fits', 'defaults',
                            'parallel', 'forgets'] + MODELS_BKT
    CV_ARGS = FIT_ARGS + ['folds', 'seed']
    DEFAULTS = {'num_fits': 5,
                'defaults': None,
                'parallel': True,
                'skills': '.*',
                'seed': random.randint(0, 1e8),
                'folds': 5,
                'forgets': False,
                'model_type': [False] * len(MODELS_BKT)}
    DEFAULTS_BKT = {'order_id': 'order_id',
                    'skill_name': 'skill_name',
                    'correct': 'correct',
                    'user_id': 'user_id',
                    'multilearn': 'template_id',
                    'multiprior': 'correct',
                    'multipair': 'problem_id',
                    'multigs': 'template_id'}
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
        self._update_param(['parallel', 'num_fits', 'seed', 'defaults'], kwargs)
        self._update_param('model_type', self._update_defaults(kwargs))

    def fit(self, data_path = None, data = None, **kwargs):
        """
        Fits a BKT model given model and data information. Takes arguments skills,
        number of initialization fits, default column names (i.e. correct, skill_name),
        parallelization, and model types.

        >>> model = Model(seed = 42)
        >>> model.fit(data_path = 'as.csv', forgets = True, skills = 'Box and Whisker')
        >>> model.evaluate(data_path = 'as.csv', metric = 'auc')
        0.6128265543747811

        """
        self._check_data(data_path, data)
        self._check_args(Model.FIT_ARGS, kwargs)
        self._update_param(['skills', 'num_fits', 'defaults', 
                            'parallel', 'forgets'], kwargs)
        self._update_param('model_type', self._update_defaults(kwargs))
        if not self.manual_param_init:
            self.fit_model = {}
        all_data = self._data_helper(data_path, data, self.defaults, self.skills, self.model_type)
        for skill in all_data:
            self.fit_model[skill] = self._fit(all_data[skill], skill, self.forgets)
        self.manual_param_init = False

    def predict(self, data_path = None, data = None):
        """
        Predicts using the trained BKT model and test data information. Takes test data
        location or DataFrame as arguments. Returns a dictionary mapping skills to predicted
        values for those skills. Note that the predicted values are a tuple of
        (correct_predictions, state_predictions).

        >>> model = Model(seed = 42)
        >>> model.fit(data_path = 'as.csv', forgets = True, skills = 'Box and Whisker')
        >>> model.predict(data_path = 'as.csv')
        {'Box and Whisker': (array([0.75036   , 0.60286553, 0.75036   , ..., 0.80416815, 0.83126012,
               0.77249206]), array([[0.17072973, 0.41991344, 0.17072973, ..., 0.07982386, 0.03405348,
                0.13333885],
               [0.82927027, 0.58008656, 0.82927027, ..., 0.92017614, 0.96594652,
                0.86666115]]))}
        """
        self._check_data(data_path, data)
        if self.fit_model is None:
            raise ValueError("model has not been fitted yet")
        all_data, df = self._data_helper(data_path = data_path, data = data,
                             defaults = self.defaults, skills = self.skills,
                             model_type = self.model_type, gs_ref = self.fit_model,
                             resource_ref = self.fit_model,
                             return_df = True)
        df['correct_predictions'] = 0
        df['state_predictions'] = 0
        for skill in all_data:
            correct_predictions, state_predictions = self._predict(self.fit_model[skill], all_data[skill])
            df.loc[all_data[skill]['index'], 'correct_predictions'] = correct_predictions
            df.loc[all_data[skill]['index'], 'state_predictions'] = state_predictions[0]
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
        if self.fit_model is None:
            raise ValueError("model has not been fitted yet")
        elif isinstance(metric, str):
            if not metric in metrics.SUPPORTED_METRICS:
                raise ValueError("metric must be one of: " + ", ".join(metrics.SUPPORTED_METRICS))
            metric = getattr(metrics, metric)
        all_data = self._data_helper(data_path, data, self.defaults, self.skills, self.model_type,
                                     gs_ref = self.fit_model, resource_ref = self.fit_model)
        return self._evaluate(all_data, metric)

    def crossvalidate(self, data = None, data_path = None, metric = metrics.rmse, **kwargs):
        """
        Crossvalidates (trains and evaluates) the BKT model. Takes the data, metric, and any
        arguments that would be passed to the fit function (skills, number of initialization fits, 
        default column names, parallelization, and model types) as arguments.

        >>> model = Model(seed = 42)
        >>> model.crossvalidate(data_path = 'as.csv', skills = 'Box and Whisker')
        {'Box and Whisker': [0.42907885779555427, 0.43025621177657264, 0.4009203965037577, 0.4198850275047665, 0.4271669758997527]}

        """
        if data is None and data_path is None:
            raise ValueError("no data specified")
        elif isinstance(metric, str):
            if not metric in metrics.SUPPORTED_METRICS:
                raise ValueError("metric must be one of: " + ", ".join(metrics.SUPPORTED_METRICS))
            metric = getattr(metrics, metric)
        self._check_args(Model.CV_ARGS, kwargs)
        self._update_param(['skills', 'num_fits', 'defaults', 
                            'parallel', 'forgets', 'seed', 'folds'], kwargs)
        self._update_param('model_type', self._update_defaults(kwargs))
        metric_vals = {}
        if not self.manual_param_init:
            self.fit_model = {}
        all_data = self._data_helper(data_path, data, self.defaults, self.skills, self.model_type)
        for skill in all_data:
            metric_vals[skill] = self._crossvalidate(all_data[skill], skill, metric)
        self.manual_param_init = False
        df = pd.DataFrame(metric_vals)
        df.columns = ['skill', 'mean_error']
        return df

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
    def coef_(self, values):
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
        """
        coefs = self.coef_
        formatted_coefs = []
        for skill in coefs:
            for param in coefs[skill]:
                classes = self._format_param(skill, param, coefs[skill][param])
                for class_ in classes:
                   formatted_coefs.append((skill, param, class_, classes[class_])) 
        df = pd.DataFrame(formatted_coefs)
        df.columns = ['skill', 'param', 'class', 'value']
        return df.set_index(['skill', 'param', 'class'])


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

    def _data_helper(self, data_path, data, defaults, skills, model_type, gs_ref = None, resource_ref = None, return_df = False):
        """ Processes data given defaults, skills, and the model type. """
        if data_path is not None:
            data_p = data_helper.convert_data(data_path, skills, defaults = defaults, model_type = model_type, 
                                              gs_refs = gs_ref, resource_refs = resource_ref, return_df = return_df)
        elif data is not None:
            data_p = data_helper.convert_data(data, skills, defaults = defaults, model_type = model_type,
                                                gs_refs = gs_ref, resource_refs = resource_ref, return_df = return_df)
        if not return_df:
            for d in data_p.values():
                check_data.check_data(d)
        else:
            for d in data_p[0].values():
                check_data.check_data(d)
        return data_p

    def _fit(self, data, skill, forgets):
        """ Helper function for fitting data. """
        num_learns = len(data["resource_names"])
        num_gs = len(data["gs_names"])
        self._check_manual_param_init(num_learns, num_gs, skill)
        num_fit_initializations = self.num_fits
        best_likelihood = float("-inf")

        for i in range(num_fit_initializations):
            fitmodel = random_model_uni.random_model_uni(num_learns, num_gs)
            if self.manual_param_init and skill in self.fit_model:
                for var in self.fit_model[skill]:
                    fitmodel[var] = self.fit_model[skill][var]

            if forgets:
                fitmodel["forgets"] = np.random.uniform(size = fitmodel["forgets"].shape)
            fitmodel, log_likelihoods = EM_fit.EM_fit(fitmodel, data, parallel = self.parallel)
            if log_likelihoods[-1] > best_likelihood:
                best_likelihood = log_likelihoods[-1]
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
        return metric(true, pred)

    def _crossvalidate(self, data, skill, metric):
        """ Helper function for crossvalidating. """
        return crossvalidate.crossvalidate(self, data, skill, self.folds, metric, self.seed)

    def _format_param(self, skill, param, value):
        """ Formats parameter for nice printing. """
        if isinstance(value, np.ndarray):
            ptype = 'resource_names' if (param == 'learns' or param == 'forgets') \
                                     else 'gs_names'
            return dict(zip(self.fit_model[skill][ptype], value))
        else:
            return {'default': value}

    def _update_param(self, params, args):
        """ Updates parameters given kwargs. """
        if isinstance(args, dict):
            for param in params:
                if param not in args:
                    setattr(self, param, Model.DEFAULTS[param])
                else:
                    setattr(self, param, args[param])
        else:
            setattr(self, params, args)

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
                                   and params[param] in Model.INITIALIZABLE_PARAMS
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
        if not data_path and not data:
            raise ValueError("no data specified")
        elif data_path is not None and data is not None:
            raise ValueError("cannot specify both data location and data")
        elif data is not None and not isinstance(data, pd.DataFrame):
            raise ValueError("data must be a Pandas DataFrame")
        elif data_path is not None and not os.path.exists(data_path):
            raise ValueError("data path is invalid or file not found")

    def __repr__(self):
        ret = 'Model('
        args = ['%s=%s' % (arg, str(getattr(self, arg))) for arg in Model.MODEL_ARGS if hasattr(self, arg)]
        ret += ', '.join(args) + ')'
        return ret
