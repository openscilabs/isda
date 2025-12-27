from setuptools import setup

setup(
    name="misda",
    version="1.0.0",
    description="MISDA: Maximal Independent Structural Dimensionality Analysis",
    author="OpenSciLabs",
    py_modules=["misda"],
    install_requires=[
        "numpy>=1.20",
        "pandas>=1.3",
        "scipy>=1.7",
        "networkx>=2.6",
        "matplotlib>=3.5",
    ],
)
