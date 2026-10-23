"""Run clang-tidy with the checks in .clang-tidy over pyBKT's C++ extension sources.

Needs clang-tidy and NumPy: python -m pip install clang-tidy numpy
Run from anywhere: python tests/run_clang_tidy.py
Exits with a non-zero status when clang-tidy reports any finding.

clang-tidy brings no omp.h, so this uses the compiler's. Set OMP_INCLUDE to
the directory that holds omp.h when gcc can't find it, for example
OMP_INCLUDE="$(brew --prefix libomp)/include" on macOS.
"""
import os
import shutil
import subprocess
import sys
import sysconfig
import tempfile
from pathlib import Path

import numpy

ROOT = Path(__file__).resolve().parent.parent
SOURCE_ROOT = ROOT / "source-cpp" / "pyBKT"
SOURCES = sorted(path for folder in ("fit", "generate") for path in (SOURCE_ROOT / folder).glob("*.cpp"))


def gcc_omp_header():
    """The path of gcc's omp.h, or None when gcc is missing or has none."""
    try:
        path = subprocess.run(["gcc", "-print-file-name=include/omp.h"], capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None
    return Path(path) if os.path.isabs(path) and Path(path).is_file() else None


def omp_include_dir(scratch):
    """A directory with omp.h in it. For gcc's, a directory holding only omp.h, so gcc's other headers stay out."""
    if os.environ.get("OMP_INCLUDE"):
        return Path(os.environ["OMP_INCLUDE"])
    header = gcc_omp_header()
    if header is None:
        sys.exit("run_clang_tidy.py: no omp.h found; install gcc or set OMP_INCLUDE to the directory with omp.h")
    (scratch / "omp.h").symlink_to(header)
    return scratch


def main():
    if shutil.which("clang-tidy") is None:
        sys.exit("run_clang_tidy.py: clang-tidy not found; install it with: python -m pip install clang-tidy")
    with tempfile.TemporaryDirectory() as scratch:
        includes = [omp_include_dir(Path(scratch)), SOURCE_ROOT / "Eigen", Path(numpy.get_include()),
                    Path(sysconfig.get_paths()["include"])]
        # -std matches setup.py.
        flags = ["-std=gnu++17", "-fopenmp", *(f"-isystem{path}" for path in includes)]
        failed = [source for source in SOURCES
                  if subprocess.run(["clang-tidy", "--quiet", str(source), "--", *flags]).returncode != 0]
    if failed:
        sys.exit(f"clang-tidy reported findings in {', '.join(str(path.relative_to(SOURCE_ROOT)) for path in failed)}")
    print(f"clang-tidy: no findings in {len(SOURCES)} files")


if __name__ == "__main__":
    main()
