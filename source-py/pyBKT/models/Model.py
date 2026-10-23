#########################################
# Model.py                              #
# Model                                 #
#                                       #
# @author Anirudhan Badrinath           #
# Last edited: 07 April 2021            #
#########################################

import numbers
import os
import pickle
import random
import urllib.request as urllib2
from collections.abc import Mapping, Sequence
from typing import (
    TYPE_CHECKING,
    Any,
    ClassVar,
    Final,
    Literal,
    TypeAlias,
    TypeGuard,
    overload,
)

import numpy as np
import pandas as pd

from pyBKT.fit import EM_fit, predict_onestep
from pyBKT.generate import random_model_uni
from pyBKT.util import check_data, crossvalidate, data_helper, metrics
from pyBKT.util.structures import (
    CoreParams,
    FixedParams,
    FixedSpec,
    FloatArray,
    InitialParams,
    NameRefs,
    Params,
    SequenceData,
    SkillData,
)

if TYPE_CHECKING:
    from typing_extensions import Unpack

    from pyBKT.util.structures import CrossvalidateOptions, FitOptions, ModelOptions

pd.options.display.float_format = '{:,.5f}'.format

# A metric, given by name or as a function, or a list or tuple of them.
MetricChoice: TypeAlias = str | metrics.Metric
MetricArgument: TypeAlias = (MetricChoice | list[str] | list[metrics.Metric] | list[MetricChoice]
                             | tuple[MetricChoice, ...])

# The array parameters coef_ accepts, in the order coef_ returns them.
_ARRAY_PARAMS: Final[tuple[Literal["learns", "guesses", "slips", "forgets"], ...]] = (
    "learns", "guesses", "slips", "forgets")


def _is_fitted(params: Params | InitialParams) -> TypeGuard[Params]:
    """Whether params came from a fit, rather than only from coef_."""
    return "As" in params


def _initial_values(params: Params | InitialParams) -> InitialParams:
    """The values coef_ reports or accepts: the prior and the array parameters present in params."""
    values: InitialParams = {}
    if "prior" in params:
        values["prior"] = params["prior"]
    for name in _ARRAY_PARAMS:
        if name in params:
            values[name] = params[name]
    return values


def _held_spec(params: Params | InitialParams) -> FixedSpec:
    """A fixed option that holds every coef_ or fitted value in params."""
    spec: FixedSpec = {}
    if "prior" in params:
        spec["prior"] = params["prior"]
    for name in _ARRAY_PARAMS:
        if name in params:
            spec[name] = params[name]
    return spec


def _seed_fit(start: Params | InitialParams, spec: FixedSpec, fitmodel: Params, fixed: FixedParams) -> None:
    """Copy a skill's coef_ or earlier fitted values into a new fit.

    A value the fixed option marks True is held; every other value becomes a starting point.
    """
    flags: Mapping[str, object] = spec
    if "prior" in start:
        if flags.get("prior") is True:
            fixed["prior"] = start["prior"]
        else:
            fitmodel["prior"] = start["prior"]
    for name in _ARRAY_PARAMS:
        if name in start:
            if flags.get(name) is True:
                fixed[name] = start[name]
            else:
                fitmodel[name] = start[name]
    if _is_fitted(start):
        for matrix in ("As", "emissions", "pi_0"):
            if flags.get(matrix) is not True:
                fitmodel[matrix] = start[matrix]


def _hold_given_values(spec: FixedSpec, fixed: FixedParams) -> None:
    """Hold each value the fixed option gives directly, rather than marking True."""
    if "prior" in spec:
        prior = spec["prior"]
        if not isinstance(prior, bool):
            fixed["prior"] = prior
    for name in _ARRAY_PARAMS:
        if name in spec:
            value = spec[name]
            if not isinstance(value, bool):
                fixed[name] = value

class Model:
    MODELS_BKT: ClassVar[list[str]] = ['multilearn', 'multiprior', 'multipair', 'multigs']
    MODEL_ARGS: ClassVar[list[str]] = ['parallel', 'num_fits', 'seed', 'defaults'] + MODELS_BKT
    FIT_ARGS: ClassVar[list[str]] = ['skills', 'num_fits', 'defaults', 'fixed',
                            'parallel', 'forgets', 'preload'] + MODELS_BKT
    CV_ARGS: ClassVar[list[str]] = FIT_ARGS + ['folds', 'seed']
    DEFAULTS: ClassVar[dict[str, Any]] = {'num_fits': 5,
                'defaults': None,
                'parallel': True,
                'skills': '.*',
                'seed': lambda: random.randint(0, int(1e8)),
                'folds': 5,
                'forgets': False,
                'fixed': None,
                'model_type': [False] * len(MODELS_BKT)}
    DEFAULTS_BKT: ClassVar[dict[str, str]] = {'order_id': 'order_id',
                    'skill_name': 'skill_name',
                    'correct': 'correct',
                    'user_id': 'user_id',
                    'multilearn': 'template_id',
                    'multiprior': 'correct',
                    'multipair': 'problem_id',
                    'multigs': 'template_id',
                    'folds': 'template_id'}
    INITIALIZABLE_PARAMS: ClassVar[list[str]] = ['prior', 'learns', 'guesses', 'slips', 'forgets']

    # Most are set with setattr in _update_param, so they are declared here for type checkers.
    fit_model: dict[str, Params | InitialParams] | None
    manual_param_init: bool
    keep: dict[str, bool]
    parallel: bool
    num_fits: int
    seed: int
    rand: np.random.RandomState
    # Column names, plus the fold count crossvalidate stores under 'folds'. convert_data adds the columns it finds.
    defaults: dict[str, Any] | None
    model_type: list[bool]
    skills: str | Sequence[str]
    folds: int | str
    forgets: bool
    fixed: bool | dict[str, FixedSpec] | None

    def __init__(self, **kwargs: "Unpack[ModelOptions]") -> None:
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

    def fit(self, data_path: str | None = None, data: pd.DataFrame | None = None,
            **kwargs: "Unpack[FitOptions]") -> None:
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

    def partial_fit(self, data_path: str | None = None, data: pd.DataFrame | None = None,
                    **kwargs: "Unpack[FitOptions]") -> None:
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
                                              preload = kwargs.get('preload', False))
        self.manual_param_init = False

    def predict(self, data_path: str | None = None, data: pd.DataFrame | None = None) -> pd.DataFrame:
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
                             model_type = self.model_type, gs_ref = self._name_refs(),
                             resource_ref = self._name_refs(),
                             return_df = True)
        # default best effort prediction of 0.5
        df['correct_predictions'] = 0.5
        df['state_predictions'] = 0.5
        for skill in all_data:
            correct_predictions, state_predictions = self._predict(self._fitted_model(skill), all_data[skill])
            state_predictions = state_predictions[1]
            multiprior_index = all_data[skill]['multiprior_index']
            if multiprior_index is not None:
                correct_predictions = np.delete(correct_predictions, multiprior_index)
                state_predictions = np.delete(state_predictions, multiprior_index)
            df.loc[all_data[skill]['index'], 'correct_predictions'] = correct_predictions
            df.loc[all_data[skill]['index'], 'state_predictions'] = state_predictions
        return df

    def evaluate(self, data: pd.DataFrame | None = None, data_path: str | None = None,
                 metric: MetricArgument = metrics.rmse) -> float | list[float]:
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
        functions, _ = self._resolve_metrics(metric)

        all_data = self._data_helper(data_path, data, self.defaults, self.skills, self.model_type,
                                     gs_ref = self._name_refs(), resource_ref = self._name_refs())
        results = self._evaluate(all_data, functions)
        return results[0] if len(results) == 1 else results

    def crossvalidate(self, data: pd.DataFrame | None = None, data_path: str | None = None,
                      metric: MetricArgument = metrics.rmse, **kwargs: "Unpack[CrossvalidateOptions]") -> pd.DataFrame:
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
        if not isinstance(data, pd.DataFrame) and not isinstance(data_path, str):
            raise TypeError("no data specified")
        functions, metric_names = self._resolve_metrics(metric)

        self._check_args(Model.CV_ARGS, kwargs)
        self._update_param(['skills', 'num_fits', 'defaults', 
                            'parallel', 'forgets', 'seed', 'folds', 'fixed'], kwargs)
        self._update_param('model_type', self._update_defaults(kwargs))
        metric_vals: dict[str, FloatArray] = {}
        if not self.manual_param_init:
            self.fit_model = {}
        if isinstance(self.folds, str):
            self._update_defaults({'folds': self.folds})
        all_data = self._data_helper(data_path, data, self.defaults, self.skills, self.model_type, folds = isinstance(self.folds, str))
        self._update_param(['skills'], {'skills': list(all_data.keys())})
        for skill in all_data:
            metric_vals[skill] = self._crossvalidate(all_data[skill], skill, functions)
        self.manual_param_init = False
        self.fit_model = {}
        df = pd.DataFrame(metric_vals.items())
        df.columns = ['skill', 'dummy']
        df[metric_names] = pd.DataFrame(df['dummy'].tolist(), index = df.index)
        return df.set_index('skill').drop(columns = 'dummy')

    @property
    def coef_(self) -> dict[str, InitialParams]:
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
        return {skill: _initial_values(params) for skill, params in self.fit_model.items()}

    @coef_.setter
    def coef_(self, values: Mapping[str, InitialParams], fixed: None = None) -> None:
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
            self.fit_model[skill] = _initial_values(values[skill])
        self.manual_param_init = True

    def params(self) -> pd.DataFrame:
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
            for param, value in coefs[skill].items():
                classes = self._format_param(skill, param, value)
                for class_ in classes:
                   formatted_coefs.append((skill, param, str(class_), classes[class_]))
        df = pd.DataFrame(formatted_coefs)
        df.columns = ['skill', 'param', 'class', 'value']
        return df.set_index(['skill', 'param', 'class'])

    def save(self, loc: str | os.PathLike[str]) -> None:
        """
        Saves a model to disk. Uses Python pickles.

        >>> model = Model(seed = 42)
        >>> model.fit(data_path = 'as.csv', multilearn = True, forgets = True, skills = 'Box and Whisker')
        >>> model.save('model.pkl')
        """
        with open(loc, 'wb') as handle:
            pickle.dump(self, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def load(self, loc: str | os.PathLike[str]) -> None:
        """
        Loads model given by loc into the current model.

        >>> model = Model(seed = 42)
        >>> model.load('model.pkl')
        """
        with open(loc, 'rb') as handle:
            orig_model = pickle.load(handle)
        for attr in vars(orig_model):
            setattr(self, attr, getattr(orig_model, attr))


    def fetch_dataset(self, link: str, loc: str) -> None:
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

    @overload
    def _data_helper(self, data_path: str | None, data: pd.DataFrame | None, defaults: dict[str, Any] | None,
                     skills: str | Sequence[str], model_type: list[bool], gs_ref: Mapping[str, NameRefs] | None = ...,
                     resource_ref: Mapping[str, NameRefs] | None = ..., return_df: Literal[False] = ...,
                     folds: bool = ...) -> dict[str, SkillData]: ...
    @overload
    def _data_helper(self, data_path: str | None, data: pd.DataFrame | None, defaults: dict[str, Any] | None,
                     skills: str | Sequence[str], model_type: list[bool], gs_ref: Mapping[str, NameRefs] | None = ...,
                     resource_ref: Mapping[str, NameRefs] | None = ..., *, return_df: Literal[True],
                     folds: bool = ...) -> tuple[dict[str, SkillData], pd.DataFrame]: ...
    def _data_helper(self, data_path: str | None, data: pd.DataFrame | None, defaults: dict[str, Any] | None,
                     skills: str | Sequence[str], model_type: list[bool], gs_ref: Mapping[str, NameRefs] | None = None,
                     resource_ref: Mapping[str, NameRefs] | None = None, return_df: bool = False,
                     folds: bool = False) -> dict[str, SkillData] | tuple[dict[str, SkillData], pd.DataFrame]:
        """ Processes data given defaults, skills, and the model type. """
        source: str | pd.DataFrame
        if isinstance(data_path, str):
            source = data_path
        elif isinstance(data, pd.DataFrame):
            source = data
        else:
            raise TypeError("no data specified")
        if return_df:
            datas, frame = data_helper.convert_data(source, skills, defaults = defaults, model_type = model_type,
                                                    gs_refs = gs_ref, resource_refs = resource_ref, return_df = True,
                                                    folds = folds)
            for d in datas.values():
                check_data.check_data(d)
            return datas, frame
        datas = data_helper.convert_data(source, skills, defaults = defaults, model_type = model_type,
                                         gs_refs = gs_ref, resource_refs = resource_ref, folds = folds)
        for d in datas.values():
            check_data.check_data(d)
        return datas

    def _fit(self, data: SkillData, skill: str, forgets: bool, preload: bool = False) -> Params:
        """ Helper function for fitting data. """
        num_learns = len(data["resource_names"])
        num_gs = len(data["gs_names"])
        self._check_manual_param_init(num_learns, num_gs, skill)
        if hasattr(self, "fixed"):
            self._check_fixed(self)
        num_fit_initializations = self.num_fits
        best_likelihood = float("-inf")
        best_model: Params | None = None
        given = self._fixed_spec(skill)

        for i in range(num_fit_initializations):
            fitmodel = random_model_uni.random_model_uni(num_learns, num_gs, rand=self.rand)
            fixed: FixedParams = {}
            if forgets:
                fitmodel["forgets"] = self.rand.uniform(size = fitmodel["forgets"].shape)
            if self.model_type[Model.MODELS_BKT.index('multiprior')]:
                fitmodel["prior"] = 0
            if self.manual_param_init and self.fit_model is not None and skill in self.fit_model:
                _seed_fit(self.fit_model[skill], self._fixed_spec(skill), fitmodel, fixed)
            _hold_given_values(given, fixed)
            if not preload:
                fitmodel, log_likelihoods = EM_fit.EM_fit(fitmodel, data, parallel = self.parallel, fixed = fixed)
                if log_likelihoods[-1] > best_likelihood:
                    best_likelihood = log_likelihoods[-1]
                    best_model = fitmodel
            else:
                best_model = fitmodel
        if best_model is None:
            raise ValueError("num_fits must be at least 1")
        fit_model = best_model
        fit_model["learns"] = fit_model["As"][:, 1, 0]
        fit_model["forgets"] = fit_model["As"][:, 0, 1]
        fit_model["prior"] = fit_model["pi_0"][1][0]
        fit_model["resource_names"] = data["resource_names"]
        fit_model["gs_names"] = data["gs_names"]
        return fit_model
    
    def _predict(self, model: CoreParams, data: SequenceData) -> tuple[FloatArray, FloatArray]:
        """ Helper function for predicting. """
        return predict_onestep.run(model, data, parallel = self.parallel)

    def _evaluate(self, all_data: Mapping[str, SequenceData], metric: Sequence[metrics.Metric]) -> list[float]:
        """ Helper function for evaluating. """
        true, pred = np.array([]), np.array([])
        for skill in all_data:
            correct_predictions, _state_predictions = self._predict(self._fitted_model(skill), all_data[skill])
            real_data = all_data[skill]['data']
            true = np.append(true, real_data.sum(axis = 0))
            pred = np.append(pred, correct_predictions)
        true = true - 1
        try:
            res = [m(true, pred) for m in metric]
        except ValueError:
            res = [m(true, pred.round(0)) for m in metric]
        return res

    def _crossvalidate(self, data: SkillData, skill: str, metric: Sequence[metrics.Metric]) -> FloatArray:
        """ Helper function for crossvalidating. """
        if isinstance(self.folds, str):
            return crossvalidate.crossvalidate(self, data, skill, self.folds, metric, self.seed, True)
        else:
            return crossvalidate.crossvalidate(self, data, skill, self.folds, metric, self.seed)

    def _format_param(self, skill: str, param: str, value: object) -> dict[str, object]:
        """ Formats parameter for nice printing. """
        if isinstance(value, np.ndarray):
            ptype: Literal['resource_names', 'gs_names'] = 'resource_names' if (param == 'learns' or param == 'forgets') \
                                     else 'gs_names'
            names = [str(i) for i in self._fitted_model(skill)[ptype]]
            return dict(sorted(zip(names, value)))
        else:
            return {'default': value}

    def _check_fixed(self, fixed: "Model") -> None:
        """ Checks fixed parameter. """
        if self.fixed is None:
            return
        elif isinstance(self.fixed, bool) and self.fixed:
            self.fixed = None if self.fit_model is None else \
                {skill: _held_spec(params) for skill, params in self.fit_model.items()}
        elif isinstance(self.fixed, dict):
            return
        else:
            raise TypeError("fixed parameter incorrectly specified")


    @overload
    def _update_param(self, params: list[str], args: Mapping[str, object], keep: bool = ...) -> None: ...
    @overload
    def _update_param(self, params: str, args: object, keep: bool = ...) -> None: ...
    def _update_param(self, params: str | list[str], args: object, keep: bool = False) -> None:
        """ Updates parameters given kwargs. """
        if isinstance(params, str):
            setattr(self, params, args)
            self.keep[params] = keep
        elif isinstance(args, Mapping):
            for param in params:
                if param not in args and (param not in self.keep or not self.keep[param]):
                    arg = Model.DEFAULTS[param]
                    setattr(self, param, arg() if callable(arg) else arg) # Allow random seed to differ between models
                elif param in args:
                    setattr(self, param, args[param])
                self.keep[param] = keep
        
        # Update RandomState if seed is one of the parameters to update
        if 'seed' in params:
            self.rand = np.random.RandomState(self.seed)

    def _update_defaults(self, defaults: Mapping[str, object]) -> list[bool]:
        """ Update the default column names. """
        model_types = [False] * 4
        for d in defaults:
            if d in Model.MODELS_BKT:
                value = defaults[d]
                if isinstance(value, bool):
                    model_types[Model.MODELS_BKT.index(d)] = value
                elif isinstance(value, str):
                    if self.defaults is None:
                        self.defaults = {}
                    self.defaults[d] = value
                    model_types[Model.MODELS_BKT.index(d)] = True
                else:
                    raise TypeError("model type must either be boolean for automatic column inference" + \
                                     " or string specifying column")
            elif d in Model.DEFAULTS_BKT:
                if self.defaults is None:
                    self.defaults = {}
                self.defaults[d] = defaults[d]
        return model_types

    def _check_params(self, params: Mapping[str, object]) -> bool:
        """ Checks if BKT parameters are valid. """
        valid = True
        for param in params:
            if param == 'prior':
                valid = valid and isinstance(params[param], numbers.Number)
            else:
                valid = valid and isinstance(params[param], np.ndarray) \
                                   and param in Model.INITIALIZABLE_PARAMS
        # The loop above has checked that every array parameter is an ndarray before valid can still be True.
        learns, forgets = params.get('learns'), params.get('forgets')
        if isinstance(learns, np.ndarray) and isinstance(forgets, np.ndarray):
            valid = valid and (len(learns) == len(forgets))
        guesses, slips = params.get('guesses'), params.get('slips')
        if isinstance(guesses, np.ndarray) and isinstance(slips, np.ndarray):
            valid = valid and (len(slips) == len(guesses))
        return valid

    def _check_manual_param_init(self, num_learns: int, num_gs: int, skill: str) -> None:
        if self.fit_model and skill in self.fit_model and 'learns' in self.fit_model[skill] \
                and len(self.fit_model[skill]['learns']) != num_learns:
            raise ValueError("invalid number of learns in initialization")
        if self.fit_model and skill in self.fit_model and 'guesses' in self.fit_model[skill] \
                and len(self.fit_model[skill]['guesses']) != num_gs:
            raise ValueError("invalid number of guess classes in initialization")
        if self.fit_model and skill in self.fit_model and 'slips' in self.fit_model[skill] \
                and len(self.fit_model[skill]['slips']) != num_gs:
            raise ValueError("invalid number of slip classes in initialization")

    def _check_args(self, expected_args: list[str], args: Mapping[str, object]) -> None:
        for arg in args:
            if arg not in expected_args:
                raise ValueError("provided arguments are not recognized. they must be one or more of: " + \
                        ", ".join(expected_args))

    def _check_data(self, data_path: str | None, data: pd.DataFrame | None) -> None:
        if not isinstance(data_path, str) and not isinstance(data, pd.DataFrame):
            raise TypeError("no data specified")
        elif isinstance(data_path, str) and isinstance(data, pd.DataFrame):
            raise TypeError("cannot specify both data location and data")
        elif isinstance(data_path, str) and not os.path.exists(data_path):
            raise ValueError("data path is invalid or file not found")

    def _fixed_spec(self, skill: str) -> FixedSpec:
        """ The fixed option's entry for skill, or an empty one. """
        return self.fixed[skill] if isinstance(self.fixed, dict) and skill in self.fixed else {}

    def _fitted_model(self, skill: str) -> Params:
        """ Returns skill's fitted model, or raises ValueError if coef_ set only its starting values. """
        if self.fit_model is None:
            raise ValueError("model has not been fitted yet")
        params = self.fit_model[skill]
        if not _is_fitted(params):
            raise ValueError("model has not been fitted yet")
        return params

    def _name_refs(self) -> dict[str, NameRefs]:
        """ The class names of every fitted skill, for convert_data to reuse. """
        if self.fit_model is None:
            return {}
        return {skill: params for skill, params in self.fit_model.items() if _is_fitted(params)}

    def _resolve_metrics(self, metric: MetricArgument) -> tuple[list[metrics.Metric], list[str]]:
        """ Returns the metric functions and their names, looking up any given by name. """
        choices = list(metric) if isinstance(metric, (tuple, list)) else [metric]
        functions: list[metrics.Metric] = []
        names: list[str] = []
        for m in choices:
            if isinstance(m, str):
                if not m in metrics.SUPPORTED_METRICS:
                    raise ValueError("metric must be one of: " + ", ".join(metrics.SUPPORTED_METRICS))
                functions.append(metrics.SUPPORTED_METRICS[m])
                names.append(m)
            elif callable(m):
                functions.append(m)
                names.append(m.__name__)
            else:
                raise TypeError("metric must either be a string, function or list/tuple of strings and functions")
        return functions, names

    def __repr__(self) -> str:
        ret = 'Model('
        args = [f'{arg}={getattr(self, arg)!s}' for arg in Model.MODEL_ARGS if hasattr(self, arg)]
        ret += ', '.join(args) + ')'
        return ret
