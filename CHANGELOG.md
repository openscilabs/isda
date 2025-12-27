<!--
SPDX-FileCopyrightText: 2025 Monaco F. J. <monaco@usp.br>
SPDX-License-Identifier: GPL-3.0-or-later
-->

# Changelog

All notable changes to the **MISDA** (Maximal Independent Structural Dimensionality Analysis) project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2025-12-26

### Rebranding
- **Project Name**: Renamed from `ISDA` to **MISDA** (Maximal Independent Structural Dimensionality Analysis).
- **Package**: `isda` -> `misda`.
- **Classes**: `ISDAResult` -> `MISDAResult`.

### Added
- **Package Structure**: extracted core logic from notebooks into a reusable `misda` Python package.
- **Installation**: added `setup.py` to support `pip install .` and remote git installation.
- **Validation**: created `benchmark.ipynb` (formerly `example.ipynb`) as a comprehensive test suite including MOP (Multi-Objective Problem) benchmarks.
- **Colab Support**: added "Open in Colab" badge and improved dependency handling for cloud execution.
- **Metrics**: Added `Component Compactness` metric to diagnose the internal consistency of identified components.
- **Micro-Metrics**: Added `Homogeneity Ratio` (min/max component correlation) and automatic Quality Warnings to detect over-reduction in transitive chains or bridge structures.
- **Diagnostics**: Added `Auto-Diagnosis` module categorizing results as 'Ideal', 'Entangled', 'Drift' (Chain), or 'Fragmented' (Bridge) based on Fidelity/Homogeneity intersection.
- **Robustness**: Added `ensure_coverage` mechanism to `isda_significance`, repairing MIS sets to guarantee statistical coverage using the execution alpha threshold.

### Changed
- **Terminology**: Renamed the fidelity metric from "LASTRO" to **SES** (Structural Evidence Score) for clarity and academic precision.
- **API**: `calculate_lastro` is now `calculate_ses`.
- **Output**: `isda_significance` now returns a dictionary with 'ses_results' where applicable.
- **Documentation**: Professionalized the introduction and structure of the validation notebook.
- **Renaming**: Renamed `validation.ipynb` to `benchmark.ipynb` to better reflect its purpose as a performance benchmark suite.

### Removed
- **Legacy Code**: Removed deprecated `explain_lastro` functions and old inline MOP definitions from the core library.
