"""Build the aqc_exec extension. Step 4.6.

    python cpp/bindings/setup.py build_ext --inplace

Or use scripts/build_cpp.sh, which puts the .pyd where Python can import it.

Requires a C++17 compiler. On Windows that means MSVC Build Tools:
    winget install --id Microsoft.VisualStudio.2022.BuildTools \
      --override "--quiet --add Microsoft.VisualStudio.Workload.VCTools --includeRecommended"
"""
from __future__ import annotations

import sys
from pathlib import Path

from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup

CPP = Path(__file__).resolve().parent.parent

extra_args = ["/O2", "/std:c++17"] if sys.platform == "win32" else ["-O3", "-std=c++17"]

ext_modules = [
    Pybind11Extension(
        "aqc_exec",
        sources=[
            str(CPP / "bindings" / "pybind_module.cpp"),
            str(CPP / "orderbook" / "orderbook.cpp"),
            str(CPP / "orderbook" / "itch_parser.cpp"),
            str(CPP / "execution" / "execution_sim.cpp"),
        ],
        include_dirs=[
            str(CPP / "include"),
            str(CPP / "orderbook"),
            str(CPP / "execution"),
        ],
        cxx_std=17,
        extra_compile_args=extra_args,
    ),
]

setup(
    name="aqc_exec",
    version="0.1.0",
    description="Order book, ITCH replay and execution simulation",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
)
