"""
Build script for larp's optional Cython extensions (larp/field/*.pyx,
larp/field/geometry/*.pyx).

Packaging metadata itself lives in pyproject.toml; this file exists only
because the Cython extensions need an explicit build step:

    python setup.py build_ext --inplace
"""
from setuptools import setup, Extension

try:
    from Cython.Build import cythonize
    import numpy as np

    module_names = ["kernels", "quadtree", "risk_field"] + [
        f"geometry.{name}" for name in ["base", "point", "linestring", "polygon", "collection"]
    ]

    ext_modules = cythonize(
        [
            Extension(
                f"larp.field.{name}",
                [f"larp/field/{name.replace('.', '/')}.pyx"],
                include_dirs=[np.get_include()],
                extra_compile_args=["-O3"],
            )
            for name in module_names
        ],
        compiler_directives={"language_level": "3"},
    )
except ImportError:
    ext_modules = []

setup(ext_modules=ext_modules)
