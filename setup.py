from setuptools import setup

setup(
    name="isda",
    version="1.0.0",
    description="ISDA: Independent Structural-Dimensionality Analysis",
    author="Codebase Refactor",
    py_modules=["isda", "mop_definitions"],
    install_requires=[
        "numpy",
        "pandas",
        "scipy",
        "networkx",
        "matplotlib",
    ],
)
