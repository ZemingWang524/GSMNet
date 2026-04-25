#!/usr/bin/env python
"""Setup script for the theta package
"""

from setuptools import setup, find_packages, Extension
from Cython.Build import cythonize
from Cython.Distutils import build_ext
import numpy

extensions = [
    Extension('series',
              sources=['series.pyx',
                       'bessel.c'],
              include_dirs=[numpy.get_include(), '/home/xxx/gsl/include'],
              library_dirs=['/home/xxx/gsl/lib'],
              libraries=['gsl', 'gslcblas'],
              extra_compile_args=["-std=c++11"],
              extra_link_args=["-std=c++11"]
              )
]

setup(
    name='functions',
    author='kruskallin',
    author_email='kruskallin@tamu.edu',
    packages=find_packages(),
    cmdclass={'build_ext': build_ext},
    ext_modules=cythonize(extensions),
    install_requires=[
        "numpy >= 1.13",
    ],
    zip_safe=False,
)
