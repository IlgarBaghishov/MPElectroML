import numpy as np
import pandas as pd
from ase import Atoms
from ase.optimize import LBFGS
from ase.filters import FrechetCellFilter
from fairchem.core import pretrained_mlip, FAIRChemCalculator
from pymatgen.core import Structure
import logging
import os
from mpelectroml.utils import HDF5_KEY_WITH_ENERGIES

logger = logging.getLogger(__name__)
_PREDICTOR = None
# PREDICTOR = pretrained_mlip.get_predict_unit("uma-sm", device="cuda")


def assign_calculator(atoms: Atoms | None) -> Atoms | None:
    """
    Assigns a FAIRChemCalculator (using 'uma-sm' model by default) to an ASE Atoms object.

    Args:
        atoms (ase.Atoms | None): The ASE Atoms object.

    Returns:
        ase.Atoms | None: The Atoms object with the calculator assigned, or None if input was None.
                          The .calc attribute will be None if calculator assignment fails.
    """
    if atoms is None:
        return None
    try:
        # Using "uma-sm". Users can check `fairchem.core.pretrained_mlip.available_models` for other options.
        # Device can be "cuda" or "cpu" depending on if a GPU is available and PyTorch is CUDA-enabled.
        model_name = "uma-sm"
        device = "cuda"
        global _PREDICTOR
        if _PREDICTOR is None:
            logger.info(f"Initializing FAIRChem predictor: {model_name} on {device}...")
            try:
                _PREDICTOR = pretrained_mlip.get_predict_unit(model_name, device=device)
                logger.info("FAIRChem predictor initialized successfully.")
            except Exception as e:
                logger.error(f"Failed to initialize FAIRChem predictor ('{model_name}'): {e}")
                _PREDICTOR = None
                return atoms
        calc = FAIRChemCalculator(_PREDICTOR, task_name="omat")
        atoms.calc = calc
        logger.debug(f"Assigned FAIRChem calculator (uma-sm) to atoms: {atoms.get_chemical_formula()}")
    except Exception as e:
        logger.error(f"Error assigning FAIRChem calculator to atoms ({atoms.get_chemical_formula()}): {e}. "
                     "Atoms object will not have a calculator.")
        atoms.calc = None  # Ensure calc is None if assignment fails
    return atoms


def relax_atoms(atoms: Atoms | None, fmax: float = 0.05, steps: int = 100) -> Atoms | None:
    """
    Relaxes an ASE Atoms object (positions and cell) using LBFGS optimizer with FrechetCellFilter.

    Args:
        atoms (ase.Atoms | None): The ASE Atoms object with a calculator already assigned.
        fmax (float): Maximum force tolerance for relaxation (eV/Angstrom).
        steps (int): Maximum number of optimization steps.

    Returns:
        ase.Atoms | None: The relaxed Atoms object, or the original if input/calculator was None or relaxation failed.
    """
    if atoms is None:
        return None
    if atoms.calc is None:
        logger.warning(f"Cannot relax atoms ({atoms.get_chemical_formula()}): No calculator assigned.")
        return atoms

    try:
        logger.debug(f"Starting relaxation for {atoms.get_chemical_formula()} with fmax={fmax}, steps={steps}.")
        optimizer = LBFGS(FrechetCellFilter(atoms))
        optimizer.run(fmax=fmax, steps=steps)
        logger.debug(f"Relaxation finished for {atoms.get_chemical_formula()}.")
    except Exception as e:
        logger.error(f"Error during LBFGS relaxation for {atoms.get_chemical_formula()}: {e}")
    return atoms


def calculate_energy_and_forces_from_Structure(
    structure_pmg: Structure | None,
    relax: bool = False,
    relax_fmax: float = 0.05,
    relax_steps: int = 100
) -> list:  # Returns [PymatgenStructure | None, energy_per_atom | None, forces_array | None]
    """
    Calculates potential energy and atomic forces for a Pymatgen Structure object
    using an ASE-compatible calculator (FAIRChem). Optionally relaxes the structure first.

    Args:
        structure_pmg (pymatgen.core.Structure | None): The input Pymatgen Structure.
        relax (bool): If True, the structure is relaxed before the final energy/force calculation.
        relax_fmax (float): Maximum force tolerance for relaxation if `relax` is True.
        relax_steps (int): Maximum steps for relaxation if `relax` is True.

    Returns:
        list: A list containing:
              - [0]: Final Pymatgen Structure (relaxed if `relax` was True, otherwise initial). None on error.
              - [1]: Calculated potential energy per atom (eV/atom). None on error.
              - [2]: NumPy array of atomic forces (eV/Angstrom). None on error.
              Returns [None, None, None] if input `structure_pmg` is None or if a
              critical error occurs during calculation.
    """
    if not isinstance(structure_pmg, Structure):
        return [None, None, None]

    atoms = structure_pmg.to_ase_atoms()
    atoms = assign_calculator(atoms)

    if atoms is None or atoms.calc is None:
        logger.warning(f"Could not assign calculator to structure: {structure_pmg.composition.reduced_formula}. "
                       "Skipping calculation.")
        return [structure_pmg, None, None]

    final_structure_output_pmg = structure_pmg
    energy_per_atom = None
    forces = None

    try:
        if relax:
            logger.info(f"Relaxing structure: {atoms.get_chemical_formula()}")
            atoms = relax_atoms(atoms, fmax=relax_fmax, steps=relax_steps)

        energy_per_atom = atoms.get_potential_energy() / len(atoms)
        forces = atoms.get_forces()

        logger.debug(f"Calculated energy/forces for {final_structure_output_pmg.composition.reduced_formula}: "
                     f"E/atom={energy_per_atom:.4f}")

    except Exception as e:
        logger.error(f"Error calculating energy/forces for {structure_pmg.composition.reduced_formula}: {e}")
        # final_structure_output_pmg remains as it was before the error
        energy_per_atom = None
        forces = None
    finally:
        if atoms and hasattr(atoms, 'calc'):
            atoms.calc = None
            final_structure_output_pmg = Structure.from_ase_atoms(atoms)

    return [final_structure_output_pmg, energy_per_atom, forces]


def add_energy_forces_to_df(
    df_pairs: pd.DataFrame,
    original_working_ion: str,  # e.g. "Li", used for naming output HDF5 file
    ion_type_to_process: str,  # "charge", "discharge", or a new_working_ion string like "Na"
    file_dirpath: str,
    idx_init: int = 0,
    idx_final: int = -1
) -> pd.DataFrame:
    """
    Adds calculated energies and forces (initial and relaxed) to the DataFrame for a specific
    type of structure (charge, discharge, or a new_working_ion's discharge).

    The function processes rows from `idx_init` to `idx_final`.
    Saves the updated DataFrame to an HDF5 file named
    `{original_working_ion}_pairs_with_energies.h5`.

    Args:
        df_pairs (pd.DataFrame): The input DataFrame. Must contain structure columns
                                 (e.g., 'charge_structure', 'Na_discharge_structure').
        original_working_ion (str): The primary working ion of the dataset (e.g., "Li"),
                                    used for naming the output HDF5 file.
        ion_type_to_process (str | None): Specifies which set of structures to process.
            - "charge": Processes 'charge_structure'.
            - "discharge": Processes 'discharge_structure'.
            - "Na", "K", etc.: Processes f"{ion_type_to_process}_discharge_structure".
            - If None, this function will log a warning and return the DataFrame unchanged.
        file_dirpath (str): Directory path to save the output HDF5 file.
        idx_init (int): Starting row index for processing.
        idx_final (int): Ending row index (exclusive) for processing. If -1, processes to the end.

    Returns:
        pd.DataFrame: The DataFrame with added/updated energy and force columns for the specified ion type.
    """
    os.makedirs(file_dirpath, exist_ok=True)
    # Output HDF5 file name is based on the original_working_ion of the dataset
    hdf5_output_path = os.path.join(file_dirpath, f"{original_working_ion}_electrode_data_with_energies.h5")

    # Determine the prefix and structure column name based on ion_type_to_process
    struct_col_name = ""
    col_prefix = ""

    if ion_type_to_process == "charge":
        struct_col_name = "charge_structure"
        col_prefix = "charge"
    elif ion_type_to_process == "discharge":
        struct_col_name = "discharge_structure"
        col_prefix = "discharge"
    else:  # Assumed to be a new working ion string like "Na", "K"
        struct_col_name = f"{ion_type_to_process}_discharge_structure"
        col_prefix = f"{ion_type_to_process}_discharge"

    if struct_col_name not in df_pairs.columns:
        logger.error(f"Structure column '{struct_col_name}' not found in DataFrame. Cannot calculate energies for "
                     f"'{ion_type_to_process}'.")
        return df_pairs

    # Define columns to be initialized or updated for this prefix
    cols_for_prefix = {
        f'{col_prefix}_init_energy_per_atom': np.nan,
        f'{col_prefix}_init_forces': pd.Series(dtype=object),  # Store as object, will hold np.ndarray or None
        f'{col_prefix}_relaxed_structure': pd.Series(dtype=object),  # Store Pymatgen Structure or None
        f'{col_prefix}_relaxed_energy_per_atom': np.nan,
        f'{col_prefix}_relaxed_forces': pd.Series(dtype=object),
    }

    for col, default_value in cols_for_prefix.items():
        if col not in df_pairs.columns:
            if isinstance(default_value, pd.Series):  # For object dtype series
                df_pairs[col] = [None] * len(df_pairs)  # Initialize with a list of Nones
                df_pairs[col] = df_pairs[col].astype(object)  # Ensure object dtype
            else:  # For float columns like energy
                df_pairs[col] = default_value  # Initialize with NaN

    if idx_final == -1:
        idx_final = df_pairs.shape[0]

    # Ensure indices are within DataFrame bounds
    idx_init = max(0, idx_init)
    idx_final = min(df_pairs.shape[0], idx_final)

    logger.info(f"Starting energy/force calculations for '{ion_type_to_process}' structures (column: "
                f"{struct_col_name}).")
    logger.info(f"Processing DataFrame rows from index {idx_init} to {idx_final - 1}.")

    for i in range(idx_init, idx_final):
        logger.info(f"Calculating for row {i} (structure type: '{ion_type_to_process}')...")

        initial_struct_pmg = df_pairs.at[i, struct_col_name]

        if not isinstance(initial_struct_pmg, Structure):
            logger.warning(f"Row {i}, {col_prefix}: Expected a Pymatgen Structure in '{struct_col_name}',"
                           f"found {type(initial_struct_pmg)}. Skipping energy calculation for this entry.")
            # Ensure corresponding energy/force columns are None/NaN for this entry
            df_pairs.at[i, f'{col_prefix}_init_energy_per_atom'] = np.nan
            df_pairs.at[i, f'{col_prefix}_init_forces'] = None
            df_pairs.at[i, f'{col_prefix}_relaxed_structure'] = None
            df_pairs.at[i, f'{col_prefix}_relaxed_energy_per_atom'] = np.nan
            df_pairs.at[i, f'{col_prefix}_relaxed_forces'] = None
            continue

        # Calculate initial energy and forces (no relaxation)
        _, init_energy, init_forces = calculate_energy_and_forces_from_Structure(initial_struct_pmg, relax=False)
        df_pairs.at[i, f'{col_prefix}_init_energy_per_atom'] = init_energy
        df_pairs.at[i, f'{col_prefix}_init_forces'] = init_forces

        # Calculate relaxed energy and forces
        relaxed_struct_pmg, relaxed_energy, relaxed_forces = calculate_energy_and_forces_from_Structure(
            initial_struct_pmg, relax=True
            )
        df_pairs.at[i, f'{col_prefix}_relaxed_structure'] = relaxed_struct_pmg
        df_pairs.at[i, f'{col_prefix}_relaxed_energy_per_atom'] = relaxed_energy
        df_pairs.at[i, f'{col_prefix}_relaxed_forces'] = relaxed_forces

        # Save intermediate results periodically
        if (i + 1) % 100 == 0:  # Save every 5 processed rows (within the current ion_type_to_process)
            logger.info(f"Processed up to row {i} for '{ion_type_to_process}', saving intermediate results "
                        f"to {hdf5_output_path}...")
            try:
                df_pairs.to_hdf(hdf5_output_path, key=HDF5_KEY_WITH_ENERGIES, mode='w')
                logger.info("Intermediate DataFrame successfully saved.")
            except Exception as e_hdf_interim:
                logger.error(f"Error saving intermediate DataFrame to HDF5: {e_hdf_interim}")

    logger.info(f"All specified rows ({idx_init} to {idx_final-1}) processed for '{ion_type_to_process}'.")
    logger.info(f"Saving final DataFrame with energies to HDF5: {hdf5_output_path} (key: {HDF5_KEY_WITH_ENERGIES})")
    try:
        df_pairs.to_hdf(hdf5_output_path, key=HDF5_KEY_WITH_ENERGIES, mode='w')
        logger.info("Final DataFrame successfully saved.")
    except Exception as e_hdf_final:
        logger.error(f"Error saving final DataFrame to HDF5: {e_hdf_final}")

    return df_pairs
