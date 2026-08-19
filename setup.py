"""
Build script for larp's optional Cython extension (larp/field_cy/kernels.pyx).

Packaging metadata itself lives in pyproject.toml; this file exists only
because the Cython extension needs an explicit build step:

    python setup.py build_ext --inplace
"""
from setuptools import setup, Extension

try:
    from Cython.Build import cythonize
    import numpy as np

    ext_modules = cythonize(
        [
            Extension(
                f"larp.field_cy.{name}",
                [f"larp/field_cy/{name}.pyx"],
                include_dirs=[np.get_include()],
                extra_compile_args=["-O3"],
            )
            for name in ["kernels", "geometry", "quadtree", "risk_field"]
        ],
        compiler_directives={"language_level": "3"},
    )
except ImportError:
    ext_modules = []

setup(ext_modules=ext_modules)
