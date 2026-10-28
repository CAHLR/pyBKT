"""Speed and memory benchmarks for pyBKT's C++ build. See benchmarks/README.md."""

import os

import pandas as pd
from pyBKT.models import Model, Roster
from pyBKT.util import data_helper

from .simulate import simulate

SIZES = [10_000, 100_000, 1_000_000]
VARIANTS = {
    "default": {},
    "forgets": {"forgets": True},
    "multigs": {"multigs": True},
    "multilearn": {"multilearn": True},
    "multiprior": {"multiprior": True},
}


def new_model():
    return Model(seed=0, num_fits=1, parallel=False)


# number = 1 gives every timed call a fresh copy from setup, because
# convert_data and fit sort the caller's frame in place.


class ConvertData:
    params = SIZES
    param_names = ["rows"]
    number = 1
    timeout = 300

    def setup(self, rows):
        self.df = simulate(rows)

    def time_convert_data(self, rows):
        data_helper.convert_data(self.df, ".*")

    def peakmem_convert_data(self, rows):
        data_helper.convert_data(self.df, ".*")


class FitBySize:
    params = SIZES
    param_names = ["rows"]
    number = 1
    timeout = 600

    def setup(self, rows):
        self.df = simulate(rows)

    def time_fit(self, rows):
        new_model().fit(data=self.df)

    def peakmem_fit(self, rows):
        new_model().fit(data=self.df)


class FitByVariant:
    params = list(VARIANTS)
    param_names = ["variant"]
    number = 1
    timeout = 300

    def setup(self, variant):
        self.df = simulate(100_000)

    def time_fit(self, variant):
        new_model().fit(data=self.df, **VARIANTS[variant])


class Predict:
    number = 1
    timeout = 300

    def setup(self):
        self.df = simulate(100_000)
        self.model = new_model()
        self.model.fit(data=self.df.copy())

    def time_predict(self):
        self.model.predict(data=self.df)


class Crossvalidate:
    number = 1
    timeout = 300

    def setup(self):
        self.df = simulate(100_000)

    def time_crossvalidate(self):
        new_model().crossvalidate(data=self.df, folds=3)


class RosterUpdates:
    def setup(self):
        df = simulate(10_000)
        model = new_model()
        model.fit(data=df, skills="skill0")
        self.students = [f"student{i}" for i in range(100)]
        self.roster = Roster(self.students, "skill0", model=model)

    def time_update_state(self):
        for student in self.students:
            self.roster.update_state("skill0", student, 1)


class Assistments:
    """ASSISTments 2009 from pyBKT-examples, when PYBKT_BENCH_ASSISTMENTS names a local copy."""

    number = 1
    timeout = 600

    def setup(self):
        path = os.environ.get("PYBKT_BENCH_ASSISTMENTS")
        if not path:
            raise NotImplementedError("set PYBKT_BENCH_ASSISTMENTS to a local as.csv")
        self.df = pd.read_csv(path, encoding="latin", low_memory=False)

    def time_convert_data(self):
        data_helper.convert_data(self.df, ".*")

    def time_fit(self):
        new_model().fit(data=self.df)
