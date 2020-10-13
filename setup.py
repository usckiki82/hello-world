"""
Further details on formatting/options for this setup.py file:
https://packaging.python.org/en/latest/distributing.html
https://docs.python-guide.org/writing/structure/
"""

from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name='kaggle-code',  # Required
    version="0.0.0",  # titanic.__version__,  # Required
    description='Code for practicing Kaggle Competitions',  # Required
    long_description=long_description,
    author='Christine Madden',  # Optional
    author_email="christine.m.madden19@gmail.com",
    packages=find_packages(),  # Required
    install_requires=[
        "setuptools",
        "wheel",
        "sklearn",
        "scipy",
        "pandas",
        "jupyter",
        "matplotlib",
        "numpy",
        "seaborn",
        "lightgbm",
        "kaggle",
        "xgboost",
        "graphviz",  # Also have to download graphviz package and add dot program folder to path
        # "tensorflow",  #need <1.19.0 on numpy
        "numba",
        # "cuPy", "cuML", "cuDF",
        # "dask",
    ]
)
