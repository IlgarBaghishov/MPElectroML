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
from chgnet.model import CHGNet, StructOptimizer

logger = logging.getLogger(__name__)
_FAIRCHEM_PREDICTOR = None
_CHGNET_PREDICTOR = None
_CHGNET_RELAXER = None


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
        global _FAIRCHEM_PREDICTOR
        if _FAIRCHEM_PREDICTOR is None:
            logger.info(f"Initializing FAIRChem predictor: {model_name} on {device}...")
            try:
                _FAIRCHEM_PREDICTOR = pretrained_mlip.get_predict_unit(model_name, device=device)
                logger.info("FAIRChem predictor initialized successfully.")
            except Exception as e:
                logger.error(f"Failed to initialize FAIRChem predictor ('{model_name}'): {e}")
                _FAIRCHEM_PREDICTOR = None
                return atoms
        calc = FAIRChemCalculator(_FAIRCHEM_PREDICTOR, task_name="omat")
        atoms.calc = calc
        logger.debug(f"Assigned FAIRChem calculator (uma-sm) to atoms: {atoms.get_chemical_formula()}")
    except Exception as e:
        logger.error(f"Error assigning FAIRChem calculator to atoms ({atoms.get_chemical_formula()}): {e}. "
                     "Atoms object will not have a calculator.")
        atoms.calc = None  # Ensure calc is None if assignment fails
    return atoms


def relax_atoms(atoms: Atoms | None, fmax: float = 0.05, steps: int = 100) -> Atoms | None:
    """
    Relaxes an ASE Atoms object using LBFGS optimizer with FrechetCellFilter.
    Specific to ASE-based calculators like UMA.

    Args:
        atoms (ase.Atoms | None): The ASE Atoms object with a calculator already assigned.
        fmax (float): Maximum force tolerance for relaxation (eV/Angstrom).
        steps (int): Maximum number of optimization steps.

    Returns:
        ase.Atoms | None: The relaxed Atoms object, or the original if input/calculator was None or relaxation failed.
    """
    if atoms is None or atoms.calc is None:
        logger.warning("Cannot relax atoms: No calculator assigned.")
        return atoms

    try:
        logger.debug(f"Starting ASE LBFGS relaxation for {atoms.get_chemical_formula()}.")
        optimizer = LBFGS(FrechetCellFilter(atoms))
        optimizer.run(fmax=fmax, steps=steps)
        logger.debug(f"Relaxation finished for {atoms.get_chemical_formula()}.")
    except Exception as e:
        logger.error(f"Error during LBFGS relaxation for {atoms.get_chemical_formula()}: {e}")
    return atoms


def calculate_energy_and_forces_from_Structure(
    structure_pmg: Structure | None,
    model_name: str = "uma",
    relax: bool = False,
    relax_fmax: float = 0.05,
    relax_steps: int = 100
) -> list:
    """
    Calculates potential energy and forces for a Pymatgen Structure.
    Can use either 'uma' (FAIRChem) or 'chgnet' model.

    Args:
        structure_pmg (pymatgen.core.Structure): The input Pymatgen Structure.
        model_name (str): The model to use, either "uma" or "chgnet".
        relax (bool): If True, relax the structure before final calculation.
        relax_fmax (float): Max force tolerance for relaxation.
        relax_steps (int): Max steps for relaxation.

    Returns:
        list: [Final Pymatgen Structure, energy_per_atom (eV/atom), forces_array (eV/A)].
              Returns [None, None, None] on critical error.
    """
    if not isinstance(structure_pmg, Structure):
        return [None, None, None]

    final_structure_pmg = structure_pmg
    energy_per_atom = None
    forces = None

    try:
        if model_name.lower() == "uma":
            try:
                atoms = structure_pmg.to_ase_atoms()
                atoms = assign_calculator(atoms)
                if atoms is None or atoms.calc is None:
                    logger.warning("Could not assign calculator to structure: "
                                   f"{structure_pmg.composition.reduced_formula}. Skipping calculation.")
                    return [structure_pmg, None, None]

                if relax:
                    logger.info(f"Relaxing structure: {atoms.get_chemical_formula()}")
                    atoms = relax_atoms(atoms, fmax=relax_fmax, steps=relax_steps)

                energy_per_atom = atoms.get_potential_energy() / len(atoms)
                forces = atoms.get_forces()

                logger.debug(f"Calculated energy/forces for {final_structure_pmg.composition.reduced_formula}: "
                             f"E/atom={energy_per_atom:.4f}")
            finally:
                if atoms and hasattr(atoms, 'calc'):
                    atoms.calc = None
                    final_structure_pmg = Structure.from_ase_atoms(atoms)

        elif model_name.lower() == "chgnet":
            try:
                global _CHGNET_PREDICTOR, _CHGNET_RELAXER
                if _CHGNET_PREDICTOR is None:
                    logger.info("Initializing CHGNet predictor...")
                    _CHGNET_PREDICTOR = CHGNet.load()
                    logger.info("CHGNet predictor initialized successfully.")

                if relax:
                    if _CHGNET_RELAXER is None:
                        logger.info("Initializing CHGNet structure relaxer...")
                        _CHGNET_RELAXER = StructOptimizer()
                        logger.info("CHGNet relaxer initialized successfully.")

                    # CHGNet relaxer takes different parameters
                    result = _CHGNET_RELAXER.relax(structure_pmg, fmax=relax_fmax, steps=relax_steps)
                    final_structure_pmg = result["final_structure"]

                # For both relaxed and unrelaxed cases, predict properties on the final structure
                prediction = _CHGNET_PREDICTOR.predict_structure(final_structure_pmg)
                energy_per_atom = prediction['e'].item()  # / len(final_structure_pmg) # CHGNet gives total energy
                forces = prediction['f']

            finally:
                if hasattr(final_structure_pmg, 'calc'):
                    final_structure_pmg.calc = None
        else:
            logger.error(f"Unknown model_name: '{model_name}'. Please use 'uma' or 'chgnet'.")
            return [structure_pmg, None, None]

        logger.debug(f"Calculated energy/forces for {final_structure_pmg.composition.reduced_formula} "
                     f"using {model_name}: E/atom={energy_per_atom:.4f}")

    except Exception as e:
        logger.error(f"Error during {model_name} calculation for {structure_pmg.composition.reduced_formula}: {e}")
        return [structure_pmg, None, None]

    return [final_structure_pmg, energy_per_atom, forces]


def add_energy_forces_to_df(
    df_pairs: pd.DataFrame,
    original_working_ion: str,
    ion_type_to_process: str,
    file_dirpath: str,
    model_name: str = "uma",  # New parameter to select model
    idx_init: int = 0,
    idx_final: int = -1
) -> pd.DataFrame:
    """
    Adds calculated energies and forces (initial and relaxed) to the DataFrame for a specific
    type of structure (charge, discharge, or a new_working_ion's discharge)
    using the specified model ('uma' or 'chgnet').

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
        model_name (str): The model to use for calculations, "uma" or "chgnet".
        idx_init (int): Starting row index for processing.
        idx_final (int): Ending row index (exclusive) for processing. If -1, processes to the end.

    Returns:
        pd.DataFrame: The DataFrame with added energy and force columns.
    """
    os.makedirs(file_dirpath, exist_ok=True)
    # Output HDF5 file name is based on the original_working_ion of the dataset
    hdf5_output_path = os.path.join(file_dirpath, f"{original_working_ion}_electrode_data_with_energies.h5")

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
        logger.error(f"Structure column '{struct_col_name}' not found in DataFrame. Cannot calculate energies"
                     f" for '{ion_type_to_process}'.")
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

    logger.info(f"Starting energy/force calculations with '{model_name}' for '{ion_type_to_process}' structures "
                f" (column: {struct_col_name}).")
    logger.info(f"Processing DataFrame rows from index {idx_init} to {idx_final - 1}.")

    for i in range(idx_init, idx_final):
        logger.info(f"Calculating for row {i} (structure type: '{ion_type_to_process}', model: '{model_name}')...")

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
        _, init_energy, init_forces = calculate_energy_and_forces_from_Structure(
            initial_struct_pmg, model_name=model_name, relax=False
        )
        df_pairs.at[i, f'{col_prefix}_init_energy_per_atom'] = init_energy
        df_pairs.at[i, f'{col_prefix}_init_forces'] = init_forces

        # Calculate relaxed energy and forces
        relaxed_struct_pmg, relaxed_energy, relaxed_forces = calculate_energy_and_forces_from_Structure(
            initial_struct_pmg, model_name=model_name, relax=True
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
