# examples/run_analysis.py
import sys
import os
import pandas as pd
import logging

# Ensure the mpelectroml package can be imported
if __name__ == '__main__':
    examples_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(examples_dir)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

from mpelectroml.data_retrieval import get_electrode_pairs, get_structures_from_electrode_pair_ids
from mpelectroml.structure_manipulation import create_new_working_ion_discharge_structures
from mpelectroml.calculations import add_energy_forces_to_df
from mpelectroml.utils import get_api_key, setup_logging, HDF5_KEY_ELECTRODE_PAIRS, HDF5_KEY_WITH_ENERGIES

# --- Configuration Variables (replaces argparse) ---
# Main settings
WORKING_ION = "Na"  # Primary working ion for initial data retrieval
NEW_WORKING_IONS = ["Li"]  # List of new working ions to substitute
FILE_DIRPATH = "."  # Directory for HDF5 data files and logs

# Workflow control flags
RESUME_FROM_FILES = False  # Attempt to resume from existing HDF5 file
SKIP_MP_RETRIEVAL = False  # Skip Materials Project data retrieval
SKIP_NEW_STRUCTURE_CREATION = False  # Skip creation of new working ion discharge structures
SKIP_CALCULATIONS = False  # Skip energy and force calculations

# Materials Project API fields to retrieve
MP_ELECTRODE_FIELDS = ["id_charge", "id_discharge"]  # for materials.insertion_electrodes.search().
# For materials.summary.search (structure and energy_per_atom are required
# formula is obtained from structure.composition.formula).
MP_SUMMARY_FIELDS = ["material_id", "structure", "energy_per_atom", "origins"]

# Calculation indexing (applies if calculations are not skipped)
CALC_TYPES = NEW_WORKING_IONS + ["charge", "discharge"]  # Types of structures to calculate energies/forces for
CALC_IDX_INITS = [0, 0, 0]  # Start index for each type of structure (must match CALC_TYPES length)
CALC_IDX_FINALS = [-1, -1, -1]  # End index for each type of structure (-1 means all remaining)

# Logging settings
LOG_LEVEL = "INFO"  # "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"
LOG_FILE_NAME = "run_analysis_simple.log"  # Name of the log file, or None to log only to console


# --- Main Execution Logic ---
def run_analysis_workflow():
    """
    Main MPElectroML workflow.
    """
    logger = logging.getLogger(__name__)  # Get logger for this script/workflow
    logger.info("Starting MPElectroML simplified workflow...")
    logger.info(f"Working Ion: {WORKING_ION}, New Ions: {NEW_WORKING_IONS}, Output Dir: {FILE_DIRPATH}")

    api_key = get_api_key()
    if not api_key and not SKIP_MP_RETRIEVAL:
        logger.error("MP_API_KEY not found. Cannot proceed with MP retrieval unless skipped.")
        return

    # Define HDF5 file paths
    base_data_hdf5_path = os.path.join(FILE_DIRPATH, f"{WORKING_ION}_electrode_data.h5")
    energies_data_hdf5_path = os.path.join(FILE_DIRPATH, f"{WORKING_ION}_electrode_data_with_energies.h5")

    df_main = pd.DataFrame()

    # Step 0: Load existing data if resuming
    if RESUME_FROM_FILES:
        if os.path.exists(energies_data_hdf5_path):
            logger.info(f"Resuming: Loading data with energies from {energies_data_hdf5_path}")
            try:
                df_main = pd.read_hdf(energies_data_hdf5_path, key=HDF5_KEY_WITH_ENERGIES)
            except Exception as e:
                logger.warning(f"Could not load energies file {energies_data_hdf5_path}: {e}. Trying base file.")
        if df_main.empty and os.path.exists(base_data_hdf5_path):
            logger.info(f"Resuming: Loading base data from {base_data_hdf5_path}")
            try:
                df_main = pd.read_hdf(base_data_hdf5_path, key=HDF5_KEY_ELECTRODE_PAIRS)
            except Exception as e:
                logger.warning(f"Could not load base data file {base_data_hdf5_path}: {e}.")
        if not df_main.empty:
            logger.info(f"Resumed with {len(df_main)} entries.")
        else:
            logger.info("No resume data found or loaded.")

    # Step 1 & 2: Get Electrode Pairs and Structures from MP
    if not SKIP_MP_RETRIEVAL:
        logger.info(f"Fetching electrode pairs for {WORKING_ION}-ion...")
        df_main = get_electrode_pairs(working_ion=WORKING_ION, api_key=api_key, fields=MP_ELECTRODE_FIELDS)

        if not df_main.empty:
            logger.info(f"Fetching structures for {len(df_main)} pairs...")
            get_structures_from_electrode_pair_ids(df_main, api_key=api_key, fields=MP_SUMMARY_FIELDS)
            try:
                df_main.to_hdf(base_data_hdf5_path, key=HDF5_KEY_ELECTRODE_PAIRS, mode='w')
                logger.info(f"Saved initial pairs and structures to {base_data_hdf5_path}")
            except Exception as e:
                logger.error(f"Error saving initial data: {e}")
        else:
            logger.error(f"Failed to retrieve electrode pairs for {WORKING_ION}. Stopping.")
            return
    elif df_main.empty:
        logger.error("MP retrieval skipped, but no data loaded via resume. Stopping.")
        return

    # Step 3: Create New Working Ion Discharge Structures
    if not SKIP_NEW_STRUCTURE_CREATION and NEW_WORKING_IONS:
        for new_ion in NEW_WORKING_IONS:
            if new_ion == WORKING_ION:
                logger.debug(f"Skipping structure creation for '{new_ion}' (same as working_ion).")
                continue

            logger.info(f"Creating new discharge structures for '{new_ion}' from '{WORKING_ION}'...")
            create_new_working_ion_discharge_structures(df_main, original_working_ion=WORKING_ION,
                                                        new_working_ion=new_ion)

        try:
            df_main.to_hdf(base_data_hdf5_path, key=HDF5_KEY_ELECTRODE_PAIRS, mode='w')
            logger.info(f"Saved updated data with new structures to {base_data_hdf5_path}")
        except Exception as e:
            logger.error(f"Error saving updated data with new structures: {e}")

    # Step 4: Add Energy and Forces
    if not SKIP_CALCULATIONS:
        logger.info("Starting energy and force calculations...")
        if len(CALC_IDX_INITS) != len(CALC_TYPES) or len(CALC_IDX_FINALS) != len(CALC_TYPES):
            logger.error("CALC_IDX_INIT and CALC_IDX_FINAL must match the length of CALC_TYPES. Please check your "
                         "configuration.")
            return

        for i, calc_type in enumerate(CALC_TYPES):
            if CALC_IDX_INITS[i] == CALC_IDX_FINALS[i]:
                logger.warning(f"Skipping {calc_type} calculations: start index equals end index "
                               f"({CALC_IDX_INITS[i]}={CALC_IDX_FINALS[i]}).")
                continue
            else:
                logger.info(f"--- Processing calculations for: {calc_type} ---")
                add_energy_forces_to_df(df_main, WORKING_ION, calc_type, FILE_DIRPATH, "uma",
                                        CALC_IDX_INITS[i], CALC_IDX_FINALS[i])
    else:
        logger.info("Skipping energy and force calculations.")

    logger.info("MPElectroML simplified workflow finished.")
    if not df_main.empty:
        logger.info(f"Final DataFrame has {len(df_main)} entries. Columns: {df_main.columns.tolist()}")
        logger.info(f"Data saved in '{FILE_DIRPATH}'")
    else:
        logger.warning("Workflow finished, but DataFrame is empty.")


if __name__ == "__main__":

    os.makedirs(FILE_DIRPATH, exist_ok=True)  # Ensure dir exists for log file

    log_file_full_path = None
    if LOG_FILE_NAME:
        log_file_full_path = os.path.join(FILE_DIRPATH, LOG_FILE_NAME)

    setup_logging(level=getattr(logging, LOG_LEVEL.upper()), log_file=log_file_full_path)

    run_analysis_workflow()
