from setuptools import setup

setup(
    name="isda",
    version="1.0.0",
    description="ISDA: Independent Structural-Dimensionality Analysis",
    author="Codebase Refactor",
    py_modules=["isda", "mop_definitions"],
    install_requires=[
        "numpy>=1.20",
        "pandas>=1.3",
        "scipy>=1.7",
        "networkx>=2.6",
        "matplotlib>=3.5",
    ],
)
