# MPElectroML Documentation

Welcome to the documentation for MPElectroML.

MPElectroML (Materials Project Electrodes with Machine-Learned Interatomic Potentials) is a Python package designed to:

1.  Retrieve electrode material data (charge/discharge pairs) from the Materials Project API.
2.  Fetch detailed structural information (Pymatgen Structure objects) for these materials.
3.  Create new candidate electrode structures by programmatically substituting working ions (e.g., replacing Li with Na, K, Mg, or Ca). This includes logic to handle supercell transformations when host frameworks require scaling for matching.
4.  Calculate potential energies and atomic forces for these original and modified structures using machine-learned interatomic potentials (MLIPs) provided by the FAIRChem library.
5.  Manage data persistence through HDF5 files, allowing for resumable workflows.

This project aims to provide a flexible and extensible tool for researchers in materials science and battery technology to automate the initial stages of computational screening and data generation for novel electrode materials.

## Key Features

-   Automated retrieval of insertion electrode data from Materials Project.
-   Generation of new electrode structures with different working ions.
-   Supercell matching logic for disparate host frameworks.
-   Energy and force calculations using FAIRChem MLIPs (e.g., 'uma-sm' model).
-   Data storage in HDF5 format.
-   Configurable logging.
-   Command-line interface for running the analysis pipeline via an example script.
-   Modular design for easier extension and integration into other workflows.

## Installation

(See `README.md` for installation instructions)

## Usage

(See `README.md` and `examples/run_analysis.py` for usage details)

## Modules Overview

-   **`mpelectroml.data_retrieval`**: Functions for fetching electrode pair IDs and their corresponding Pymatgen structures from the Materials Project.
-   **`mpelectroml.structure_manipulation`**: Functions for creating new Pymatgen `Structure` objects by substituting working ions, including supercell generation and host framework matching logic.
-   **`mpelectroml.calculations`**: Functions for assigning MLIP calculators (FAIRChem), performing structural relaxations (ASE LBFGS), and calculating energies/forces.
-   **`mpelectroml.utils`**: Helper utilities for API key management, logging setup, and common constants.

---

*This documentation is a basic placeholder. For detailed API references, please refer to the docstrings within the source code.*