# MPElectroML

**MPElectroML: Materials Project Electrodes with Machine-Learned Interatomic Potentials**

MPElectroML is a Python package designed to automate the process of fetching electrode material data from the Materials Project, performing structural modifications (such as substituting working ions like Li with Na, K, Mg, or Ca), and subsequently calculating energies and forces for these structures using machine-learned interatomic potentials (MLIPs) via the FAIRChem library. This tool aims to assist researchers in materials science and battery technology by streamlining data collection for novel electrode materials discovery.

The project facilitates the exploration of how different working ions might perform in known host structures, leveraging the vast database of the Materials Project and the predictive power of modern MLIPs for rapid property evaluation.

## Key Features

-   **Automated Data Retrieval**: Fetches insertion electrode data (charge/discharge pairs) and detailed structural information from the Materials Project using its official API.
-   **Structural Modification**: Programmatically substitutes working ions in electrode structures. For instance, it can generate sodiated, potassiated, etc., versions from known lithiated materials.
-   **MLIP Calculations**: Performs structural relaxation and calculates potential energies and atomic forces using FAIRChem MLIPs (e.g., the 'uma-sm' model is used by default).
-   **Data Persistence**: Saves processed data, including Pymatgen Structure objects and calculated properties, in HDF5 format for easy access, sharing, and analysis.
-   **Resumable Workflow**: Allows interruption and resumption of lengthy data processing or calculation steps.
-   **Modular Design**: Core functionalities are separated into modules for data retrieval, structure manipulation, and calculations, facilitating easier extension and integration.

## Installation

### Prerequisites

-   Python 3.10 or higher.
-   An active Materials Project API key. You need to set this as an environment variable:
    ```bash
    export MP_API_KEY="YOUR_MP_API_KEY"
    ```
    Replace `"YOUR_MP_API_KEY"` with your actual key.

### Steps

1.  **Clone the repository (optional, for development or direct use):**
    ```bash
    git clone [https://github.com/IlgarBaghishov/MPElectroML.git](https://github.com/IlgarBaghishov/MPElectroML.git)
    cd MPElectroML
    ```

2.  **Install the package:**

    It is recommended to use a virtual environment:

    ```bash
    python -m venv venv_mpelectroml
    source venv_mpelectroml/bin/activate  # On Linux/macOS
    # venv_mpelectroml\Scripts\activate    # On Windows
    ```

    Alternatively, you can use Conda to manage environments.

    **For GPU Support (Recommended for `fairchem` performance):**

    If you want to use GPUs for the FAIRChem model, we recommend to install a GPU-compatible version of PyTorch **before** installing `MPElectroML`. This ensures that `fairchem-core` picks up the correct PyTorch installation.

      * At the time of writing (June 3, 2025), FAIRChem is known to work well with PyTorch version 2.6.0.
      * First, determine your system's CUDA version (e.g., by running `nvidia-smi`).
      * Then, visit the [PyTorch website](https://pytorch.org/get-started/locally/) or the [PyTorch previous versions page](https://pytorch.org/get-started/previous-versions/) to find the appropriate installation command. For example, to install PyTorch 2.6.0 with support for CUDA 12.6, the command would be:
        ```bash
        pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu126
        ```
        (Ensure you adjust `cu126` and the index URL if your CUDA version is different, and verify that `torch==2.6.0` is the required version for FAIRChem.)

    **Then, install MPElectroML:**

      * For editable mode (if you cloned the repo and want to modify the code):
        ```bash
        pip install -e .
        ```
      * To include development and testing dependencies:
        ```bash
        pip install -e .[dev]
        ```
      * For a standard install (e.g., if installing from PyPI in the future, or from a wheel file):
        ```bash
        pip install . 
        # or pip install mpelectroml (once published)
        ```

    The FAIRChem library (`fairchem.core`) and its models might have other specific installation requirements. Please refer to the [FAIRChem documentation](https://github.com/FAIRChemistry/fairchem) for details if you encounter issues related to it.

## Usage

The primary way to use MPElectroML is through the example script `examples/run_analysis.py`.

### Running the Analysis Pipeline

Navigate to the `examples` directory and run the script:
```bash
cd examples
python run_analysis.py
```

### Using MPElectroML as a Library

You can also import and use the functions from the `mpelectroml` package in your own Python scripts:

```python
import os
from mpelectroml.utils import get_api_key, setup_logging
from mpelectroml.data_retrieval import get_electrode_pairs, get_structures_from_electrode_pair_ids
from mpelectroml.structure_manipulation import create_new_working_ion_discharge_structures
from mpelectroml.calculations import add_energy_forces_to_df

# Setup
setup_logging(level="DEBUG") # Example logging setup
api_key = get_api_key()
output_dir = "custom_mpelectroml_run"
os.makedirs(output_dir, exist_ok=True)

# 1. Get Li-ion electrode pairs
print("Fetching Li-ion pairs...")
df_li_pairs = get_electrode_pairs(working_ion="Li", file_dirpath=output_dir, api_key=api_key)

if df_li_pairs is not None and not df_li_pairs.empty:
    # 2. Get structures for these pairs
    print("Fetching structures for Li-ion pairs...")
    get_structures_from_electrode_pair_ids(df_li_pairs, api_key=api_key)
    
    # 3. Create Na-ion discharge structures from Li-ion ones
    print("Creating Na-ion discharge structures...")
    create_new_working_ion_discharge_structures(
        df_pairs=df_li_pairs,
        original_working_ion="Li",
        new_working_ion="Na"
    )
    
    # 4. Calculate energies for Li charge/discharge and Na discharge
    print("Calculating energies for Li charge structures...")
    df_li_pairs = add_energy_forces_to_df(
        df_pairs=df_li_pairs, original_working_ion="Li", 
        ion_type_to_process="charge", file_dirpath=output_dir
    )
    print("Calculating energies for Li discharge structures...")
    df_li_pairs = add_energy_forces_to_df(
        df_pairs=df_li_pairs, original_working_ion="Li",
        ion_type_to_process="discharge", file_dirpath=output_dir
    )
    print("Calculating energies for Na discharge structures...")
    df_li_pairs = add_energy_forces_to_df(
        df_pairs=df_li_pairs, original_working_ion="Li", # Still based on original Li dataset for file naming
        ion_type_to_process="Na", file_dirpath=output_dir
    )
    
    print("\nFinal DataFrame head:")
    print(df_li_pairs.head())
    print(f"\nData saved in HDF5 files in '{output_dir}'")
else:
    print("Failed to retrieve Li-ion pairs.")
```

## Data Output

The pipeline primarily generates HDF5 files in the specified output directory:

-   `{working_ion}_electrode_data.h5`: (Key: defined in `mpelectroml.utils.HDF5_KEY_ELECTRODE_PAIRS`)
    Stores initial electrode pair IDs, their Pymatgen structure objects, formulas, MP energies, and any new ion structures created by substitution.
-   `{working_ion}_electrode_data_with_energies.h5`: (Key: defined in `mpelectroml.utils.HDF5_KEY_WITH_ENERGIES`)
    Contains all data from the above file, plus MLIP-calculated initial and relaxed energies/forces for all relevant structures.

### Understanding the `{working_ion}_electrode_data_with_energies.h5` DataFrame

The `{working_ion}_electrode_data_with_energies.h5` file (where `{working_ion}` is the primary ion like "Li", and the HDF5 key is typically `data_with_energies`) stores the main pandas DataFrame resulting from the MPElectroML pipeline. This DataFrame accumulates data from all processing steps and includes the following columns:

**1. Base Electrode Pair Information (from Materials Project):**

* `charge_id`: (Text) The Materials Project ID (e.g., "mp-xxxxx") of the charged electrode material.
* `discharge_id`: (Text) The Materials Project ID of the discharged electrode material (containing the original `working_ion`).

**2. Original Working Ion Structures and Properties (from Materials Project):**

* `charge_structure`: (Pymatgen `Structure` object) The crystal structure of the charged material, as obtained from Materials Project.
* `charge_formula`: (Text) The chemical formula of the charged material (e.g., "CoO2").
* `charge_energy_per_atom`: (Float) The energy per atom (in eV/atom) of the charged material, as reported by Materials Project. **Note:** This is DFT-calculated energy from MP, not from the MLIP.
* `discharge_structure`: (Pymatgen `Structure` object) The crystal structure of the discharged material (containing the original `working_ion`), as obtained from Materials Project.
* `discharge_formula`: (Text) The chemical formula of the discharged material (e.g., "LiCoO2").
* `discharge_energy_per_atom`: (Float) The energy per atom (in eV/atom) of the discharged material, as reported by Materials Project. **Note:** This is DFT-calculated energy from MP, not from the MLIP.

**3. Generated Structures for New Working Ions:**

For each `new_working_ion` (e.g., "Na", "K") specified in the workflow, columns are generated:
* `{new_working_ion}_discharge_structure`: (Pymatgen `Structure` object) The crystal structure of the discharged host material after substituting the original `working_ion` with the `new_working_ion`. For example, `Na_discharge_structure`.
* `{new_working_ion}_discharge_formula`: (Text) The chemical formula corresponding to the `{new_working_ion}_discharge_structure`. For example, `Na_discharge_formula`.

**4. MLIP-Calculated Properties:**

For each type of structure processed by the MLIP calculations (e.g., "charge", "discharge", and each "new\_working\_ion\_discharge" like "Na\_discharge"), a set of columns is added with a corresponding prefix. Let `{prefix}` represent these identifiers (e.g., `charge`, `discharge`, `Na_discharge`).

* `{prefix}_init_energy_per_atom`: (Float) The potential energy per atom (in eV/atom) of the *initial* (unrelaxed by MLIP) `{prefix}_structure`, as calculated by the FAIRChem MLIP.
* `{prefix}_init_forces`: (NumPy array object) An array where each row contains the [fx, fy, fz] force components (in eV/Å) acting on each atom in the *initial* `{prefix}_structure`, calculated by the MLIP.
* `{prefix}_relaxed_structure`: (Pymatgen `Structure` object) The crystal structure obtained after performing structural relaxation (positions and cell) on the `{prefix}_structure` using the FAIRChem MLIP.
* `{prefix}_relaxed_energy_per_atom`: (Float) The potential energy per atom (in eV/atom) of the *MLIP-relaxed* `{prefix}_structure`.
* `{prefix}_relaxed_forces`: (NumPy array object) An array of atomic forces (in eV/Å) for the *MLIP-relaxed* `{prefix}_structure`. If relaxation converged well, these forces should be close to zero.

**Example Column Names for MLIP Data (if `WORKING_ION="Li"` and `NEW_WORKING_IONS=["Na"]`):**

* For the original charged structure: `charge_init_energy_per_atom`, `charge_relaxed_structure`, etc.
* For the original discharged structure (e.g., lithiated): `discharge_init_energy_per_atom`, `discharge_relaxed_structure`, etc.
* For the new ion discharged structure (e.g., sodiated): `Na_discharge_init_energy_per_atom`, `Na_discharge_relaxed_structure`, etc.

This comprehensive DataFrame allows for direct comparison of MP-DFT energies with MLIP-calculated energies, analysis of structural changes upon relaxation with the MLIP, and evaluation of different working ions.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details. TODO: Update if you choose a different license.
