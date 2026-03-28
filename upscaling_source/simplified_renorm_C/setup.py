from setuptools import setup, Extension
import os
import pybind11

ext_modules = [
    Extension(
        "simplified_renorm_C",
        ["bindings.cpp", "simplified_renorm.cpp"],
        include_dirs=[pybind11.get_include(), os.path.abspath(".")],
        language="c++",
    ),
]

setup(
    name="simplified_renorm_C",
    ext_modules=ext_modules,
)