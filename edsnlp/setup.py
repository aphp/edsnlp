#!/usr/bin/env python
from distutils.sysconfig import get_python_inc

import numpy
from setuptools import Extension, setup

# See if Cython is installed
try:
    from Cython.Build import cythonize
    from Cython.Distutils import build_ext
except ImportError:
    # Do nothing if Cython is not available
    # Got to provide this function. Otherwise, poetry will fail
    print("You must install Cython to build this library")
else:
    # Cython is installed. Compile
    print("Compiling")

    COMPILER_DIRECTIVES = {
        "language_level": "3",
    }
    MOD_NAMES = [
        "edsnlp.matchers.phrase",
        "edsnlp.pipelines.core.sentences.sentences",
    ]

    include_dirs = [
        numpy.get_include(),
        get_python_inc(plat_specific=True),
    ]
    ext_modules = []
    for name in MOD_NAMES:
        mod_path = name.replace(".", "/") + ".pyx"
        ext = Extension(
            name,
            [mod_path],
            language="c++",
            include_dirs=include_dirs,
            extra_compile_args=["-std=c++11"],
        )
        ext_modules.append(ext)
    print("Cythonizing sources")
    ext_modules = cythonize(ext_modules, compiler_directives=COMPILER_DIRECTIVES)

    setup(
        ext_modules=ext_modules,
        package_data={
            "": ["*.dylib", "*.so"],
        },
        cmdclass={"build_ext": build_ext},
    )
