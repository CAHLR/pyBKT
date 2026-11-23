# Benchmarks

These [asv](https://asv.readthedocs.io/) benchmarks time pyBKT's C++ build on seeded simulated data at 10,000, 100,000 and 1,000,000 rows. They cover `convert_data`, `fit` for each model variant, `predict`, `crossvalidate` and `Roster.update_state`, plus peak memory for `convert_data` and `fit`. `Assistments` times `convert_data` and `fit` on ASSISTments 2009 when `PYBKT_BENCH_ASSISTMENTS` names a local copy of `as.csv` from [pyBKT-examples](https://github.com/CAHLR/pyBKT-examples/tree/master/data).

Every pull request runs `.github/workflows/benchmarks.yml`. It times the base and the head on the same runner and lists anything more than 1.2 times slower in the job summary. It doesn't fail the pull request.

## Run locally

```sh
python -m pip install asv virtualenv
asv machine --yes
asv continuous --factor 1.2 master HEAD
```

`asv run --quick` runs each benchmark once, to check that they work. Results go to `.asv/`, which git ignores.

## Pure Python

asv builds pyBKT with `PYBKT_REQUIRE_CPP=1`, so it times only the C++ build. To time the pure Python build, run the benchmarks in an environment without pyBKT installed. asv ignores `PYTHONPATH`, so point `ASV_PYTHONPATH` at `source-py`:

```sh
python -m pip install asv numpy pandas requests
ASV_PYTHONPATH="$PWD/source-py" asv run --environment existing:python --quick --bench "ConvertData|FitByVariant"
```

The pure Python E-step is much slower, so leave out `FitBySize`.
