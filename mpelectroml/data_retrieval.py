# mpelectroml/data_retrieval.py
import os
import time
import pandas as pd
from mp_api.client import MPRester
import logging

# Configure logger for this module
logger = logging.getLogger(__name__)


def get_electrode_pairs(working_ion: str, file_dirpath: str, api_key: str | None) -> pd.DataFrame:
    """
    Retrieves electrode pairs data (charge and discharge IDs) from the Materials Project
    for a given working ion. Saves the data to an HDF5 file.

    Args:
        working_ion (str): The working ion (e.g., "Li", "Na").
        file_dirpath (str): Directory path to save the HDF5 file.
                            The filename will be f"{working_ion}_electrode_data.h5".
        api_key (str | None): Materials Project API key.

    Returns:
        pd.DataFrame: DataFrame containing 'charge_id' and 'discharge_id' for electrode pairs.
                      Returns an empty DataFrame if an error occurs or no data is found.
    """
    # Ensure the output directory exists
    os.makedirs(file_dirpath, exist_ok=True)
    hdf5_file_path = os.path.join(file_dirpath, f"{working_ion}_electrode_data.h5")
    hdf5_key = "data"

    found_pairs_data = []
    retrieved_count = 0
    start_time = time.time()
    df_found_pairs = pd.DataFrame()

    try:
        with MPRester(api_key=api_key) as mpr:
            logger.debug("Available insertion electrode fields: "
                         f"{mpr.materials.insertion_electrodes.available_fields}")
            logger.info(f"Querying Materials Project for {working_ion}-ion insertion electrodes...")

            electrode_docs = mpr.materials.insertion_electrodes.search(
                working_ion=working_ion,
                fields=["id_charge", "id_discharge"]
            )
            retrieved_count = len(electrode_docs)
            logger.info(f"Retrieved {retrieved_count} initial insertion electrode entries from MP.")

            for doc in electrode_docs:
                if doc.id_charge and doc.id_discharge:  # Ensure both IDs are present
                    found_pairs_data.append({
                        "charge_id": doc.id_charge,
                        "discharge_id": doc.id_discharge,
                    })
                else:
                    logger.debug(f"Skipping entry due to missing charge/discharge ID: {doc}")

        if found_pairs_data:
            df_found_pairs = pd.DataFrame(found_pairs_data)
            logger.info(f"Found {len(df_found_pairs)} unique electrode pairs.")
            try:
                df_found_pairs.to_hdf(hdf5_file_path, key=hdf5_key, mode='w')
                logger.info(f"Electrode pairs DataFrame successfully saved to {hdf5_file_path} (key: {hdf5_key})")
            except Exception as e_hdf:
                logger.error(f"Error saving DataFrame to HDF5 file {hdf5_file_path}: {e_hdf}")
        else:
            logger.warning(f"No electrode pairs data retrieved or processed for {working_ion}-ion.")

    except Exception as e:
        logger.error(f"An error occurred during electrode pair retrieval: {e}")
        if hasattr(e, 'response') and e.response is not None:
            logger.error(f"API Response Status: {e.response.status_code}")
            logger.error(f"API Response Text: {e.response.text}")
        if "401" in str(e) or "403" in str(e):
            logger.error("Please verify your Materials Project API key is correct, active, "
                         "and has necessary permissions.")

    finally:
        end_time = time.time()
        logger.info(f"Electrode pair retrieval from MP finished in {end_time - start_time:.2f} seconds.")

    return df_found_pairs


def get_structures_from_electrode_pair_ids(df_pairs: pd.DataFrame, api_key: str | None) -> None:
    """
    Retrieves Pymatgen Structure objects and related information (formula, energy per atom)
    for charge and discharge IDs present in the input DataFrame.
    Modifies the input DataFrame `df_pairs` in place by adding new columns:
    'charge_structure', 'charge_formula', 'charge_energy_per_atom',
    'discharge_structure', 'discharge_formula', 'discharge_energy_per_atom'.

    Args:
        df_pairs (pd.DataFrame): DataFrame containing 'charge_id' and 'discharge_id' columns.
                                 This DataFrame will be modified.
        api_key (str | None): Materials Project API key.
    """
    if df_pairs.empty:
        logger.warning("Input DataFrame `df_pairs` is empty. Skipping structure retrieval.")
        return

    # Get unique material IDs to minimize API calls
    charge_ids = df_pairs['charge_id'].dropna().unique().tolist()
    discharge_ids = df_pairs['discharge_id'].dropna().unique().tolist()
    all_unique_ids = list(set(charge_ids + discharge_ids))

    if not all_unique_ids:
        logger.info("No valid material IDs found in DataFrame. Skipping structure retrieval.")
        return

    logger.info(f"Retrieving structures for {len(all_unique_ids)} unique material IDs...")
    start_time = time.time()

    material_docs_map = {}
    try:
        with MPRester(api_key=api_key) as mpr:
            # Fields to retrieve from the summary endpoint
            requested_fields = ["material_id", "structure", "energy_per_atom", "formula_pretty"]

            # Fetch all unique material docs in one go if possible, or chunk if too many
            # For simplicity, fetching all at once here. Consider chunking for > few thousands IDs.
            retrieved_docs = mpr.materials.summary.search(material_ids=all_unique_ids, fields=requested_fields)

            for doc in retrieved_docs:
                material_docs_map[doc.material_id] = doc
            logger.info(f"Retrieved data for {len(material_docs_map)} out of {len(all_unique_ids)} "
                        "requested material IDs.")

    except Exception as e:
        logger.error(f"An error occurred during structure retrieval from MP: {e}")
        if hasattr(e, 'response') and e.response is not None:
            logger.error(f"API Response Status: {e.response.status_code}")
            logger.error(f"API Response Text: {e.response.text}")
        return  # Exit if API call fails

    # Helper function to map data
    def map_structure_data(material_id, docs_map):
        doc = docs_map.get(material_id)
        if doc and doc.structure:
            return doc.structure, doc.structure.composition.formula, doc.energy_per_atom
        elif doc:  # Doc exists but no structure
            logger.warning(f"Material ID {material_id} found but structure is missing in MP data.")
            return None, doc.formula_pretty if doc.formula_pretty else None, doc.energy_per_atom
        else:  # Doc not found for this ID
            logger.warning(f"Material ID {material_id} not found in retrieved MP data.")
            return None, None, None

    # Apply mapping to DataFrame columns
    charge_data = df_pairs['charge_id'].apply(lambda mid: pd.Series(map_structure_data(mid, material_docs_map),
                                                                    index=['structure', 'formula', 'energy_per_atom']))
    df_pairs[['charge_structure', 'charge_formula', 'charge_energy_per_atom']] = charge_data

    discharge_data = df_pairs['discharge_id'].apply(lambda mid:
                                                    pd.Series(map_structure_data(mid, material_docs_map),
                                                              index=['structure', 'formula', 'energy_per_atom']))
    df_pairs[['discharge_structure', 'discharge_formula', 'discharge_energy_per_atom']] = discharge_data

    end_time = time.time()
    logger.info(f"Finished retrieving and adding structures to DataFrame in {end_time - start_time:.2f} seconds.")
    # Log count of successfully retrieved structures
    logger.info(f"Successfully added {df_pairs['charge_structure'].notna().sum()} charge structures.")
    logger.info(f"Successfully added {df_pairs['discharge_structure'].notna().sum()} discharge structures.")
