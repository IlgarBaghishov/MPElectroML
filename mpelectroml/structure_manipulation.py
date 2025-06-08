import numpy as np
import pandas as pd
from pymatgen.core import Structure, Element, Composition
from pymatgen.analysis.structure_matcher import StructureMatcher
import logging

logger = logging.getLogger(__name__)


def generate_multiples(n: int) -> list[list[int]]:
    """
    Generates all ordered combinations of three positive integers (a, b, c)
    such that a * b * c = n. This is used for determining possible supercell
    transformations.

    Args:
        n (int): The integer for which to find multiplicative factors.
                 Must be a positive integer.

    Returns:
        list[list[int]]: A list of [a, b, c] triplets.

    Raises:
        ValueError: If n is not a positive integer.
    """
    if not isinstance(n, int) or n <= 0:
        raise ValueError("Input n must be a positive integer for generate_multiples.")

    results = []
    for a in range(1, n + 1):
        if n % a == 0:
            remaining_n_div_a = n // a
            for b in range(1, remaining_n_div_a + 1):
                if remaining_n_div_a % b == 0:
                    c = remaining_n_div_a // b
                    results.append([a, b, c])
    return results


def create_new_working_ion_discharge_structures(
    df_pairs: pd.DataFrame,
    original_working_ion: str = "Li",
    new_working_ion: str = "Na"
) -> None:
    """
    Creates new discharge Pymatgen Structure objects by replacing the `original_working_ion`
    with the `new_working_ion` in the 'discharge_structure' column.
    This function attempts to handle cases where the host frameworks (after removing
    working ions) of the charge and discharge structures do not initially match in size,
    by applying supercell transformations.

    Modifies the input DataFrame `df_pairs` in place by adding columns:
    - f"{new_working_ion}_discharge_structure" (Pymatgen Structure objects or None)
    - f"{new_working_ion}_discharge_formula" (string chemical formulas or None)

    Args:
        df_pairs (pd.DataFrame): DataFrame which must contain 'charge_structure' and
                                 'discharge_structure' columns with Pymatgen Structure objects.
        original_working_ion (str): The chemical symbol of the working ion to be replaced (e.g., "Li").
        new_working_ion (str): The chemical symbol of the new working ion to substitute (e.g., "Na").
    """
    new_struct_col_name = f"{new_working_ion}_discharge_structure"
    new_formula_col_name = f"{new_working_ion}_discharge_formula"

    # Initialize new columns with None if they don't exist
    if new_struct_col_name not in df_pairs.columns:
        df_pairs[new_struct_col_name] = pd.Series([None] * len(df_pairs), dtype=object)
    if new_formula_col_name not in df_pairs.columns:
        df_pairs[new_formula_col_name] = pd.Series([None] * len(df_pairs), dtype=object)

    new_discharge_structure_list = []
    match_stats = {"unmatched_complex": 0, "original_ion_in_charge_host": 0, "processed_rows": 0}

    for i in range(df_pairs.shape[0]):
        match_stats["processed_rows"] += 1
        charge_struct_pmg = df_pairs.at[i, 'charge_structure']
        discharge_struct_pmg = df_pairs.at[i, 'discharge_structure']

        # Ensure we have Pymatgen Structure objects to work with
        if not isinstance(charge_struct_pmg, Structure) or not isinstance(discharge_struct_pmg, Structure):
            logger.debug(f"Row {i}: Skipping due to missing Pymatgen charge or discharge structure.")
            new_discharge_structure_list.append(None)
            continue

        charge_struct_pmg_copy = charge_struct_pmg.copy()
        discharge_struct_pmg_copy = discharge_struct_pmg.copy()
        new_discharge_structure = discharge_struct_pmg.copy()

        if original_working_ion in Composition(charge_struct_pmg.formula).get_el_amt_dict():
            match_stats["original_ion_in_charge_host"] += 1

            # Create host frameworks by removing the original working ion
            charge_host = charge_struct_pmg_copy.copy()
            charge_host.remove_species([original_working_ion])

            discharge_host = discharge_struct_pmg_copy.copy()
            discharge_host.remove_species([original_working_ion])

            if charge_host.composition.formula != discharge_host.composition.formula:
                logger.debug(f"Row {i}: Host formulas differ for {original_working_ion} removal. "
                             f"Charge host: {charge_host.composition.formula}, "
                             f"Discharge host: {discharge_host.composition.formula}. Attempting supercell matching.")

                len_charge_host = len(charge_host)
                len_discharge_host = len(discharge_host)

                scaling_ratio = 0
                smaller_structure = None
                larger_structure = None

                if len_discharge_host % len_charge_host == 0:
                    scaling_ratio = round(len_discharge_host / len_charge_host)
                    smaller_structure = charge_host
                    larger_structure = discharge_host
                elif len_charge_host % len_discharge_host == 0:
                    scaling_ratio = round(len_charge_host / len_discharge_host)
                    smaller_structure = discharge_host
                    larger_structure = charge_host

                if scaling_ratio > 0 and smaller_structure and larger_structure:
                    na_nb_nc_options = generate_multiples(scaling_ratio)

                    smaller_lattice = smaller_structure.lattice
                    larger_lattice = larger_structure.lattice

                    na = larger_lattice.a / smaller_lattice.a
                    nb = larger_lattice.b / smaller_lattice.b
                    nc = larger_lattice.c / smaller_lattice.c

                    closest_index = np.argmin(np.linalg.norm(np.array(na_nb_nc_options) -
                                                             np.array([na, nb, nc]), axis=1))
                    na, nb, nc = na_nb_nc_options[closest_index]

                    if len_charge_host < len_discharge_host:
                        charge_struct_pmg_copy = charge_struct_pmg_copy.make_supercell([na, nb, nc])
                    else:
                        discharge_struct_pmg_copy = discharge_struct_pmg_copy.make_supercell([na, nb, nc])

                else:
                    logger.warning(f"Row {i}: Atom number ratio for hosts is not an integer multiple or one host is "
                                   "empty. Cannot scale for matching.")
                    new_discharge_structure_list.append(None)
                    continue

            matcher = StructureMatcher(ltol=0.6, stol=0.8, angle_tol=20, primitive_cell=False,
                                       scale=False, allow_subset=True)
            mapping = matcher.get_mapping(discharge_struct_pmg_copy, charge_struct_pmg_copy)

            new_discharge_structure = discharge_struct_pmg_copy.copy()

            if mapping is None:
                logger.warning(f"Row {i}: Structures do not match after scaling attempt for {original_working_ion}. "
                               "No replacement will be made.")
                match_stats["unmatched_complex"] += 1
                new_discharge_structure = None
            else:
                for j, site in enumerate(new_discharge_structure):
                    if j not in mapping:
                        if Element(original_working_ion) in site:
                            new_discharge_structure.replace(j, new_working_ion)
                        else:
                            logger.warning(f"Row {i}: Site {j} in discharge structure does not match any site in "
                                           f"charge structure and it is not {original_working_ion} but "
                                           f"{site.species_string}.")
        else:
            logger.debug(f"Row {i}: '{original_working_ion}' not in charge formula. Attempting simple replacement in "
                         f"discharge structure for '{new_working_ion}'.")
            if original_working_ion in Composition(new_discharge_structure.formula).get_el_amt_dict():
                new_discharge_structure.replace_species({original_working_ion: new_working_ion})
                new_discharge_structure = new_discharge_structure
            else:
                logger.warning(f"Row {i}: Simple replacement: '{original_working_ion}' not found in discharge formula "
                               f"'{new_discharge_structure.formula}'. Cannot create '{new_working_ion}' variant. "
                               "Structure set to None.")
                new_discharge_structure = None

        new_discharge_structure_list.append(new_discharge_structure)

    df_pairs[new_struct_col_name] = new_discharge_structure_list
    df_pairs[new_formula_col_name] = [s.composition.formula if isinstance(s, Structure) else None
                                      for s in new_discharge_structure_list]

    logger.info(f"Finished creating new discharge structures for ion: '{new_working_ion}'.")
    logger.info(f"Stats for '{new_working_ion}': "
                f"Total rows processed: {match_stats['processed_rows']}. Rows where original ion "
                f"('{original_working_ion}') was in charge host: {match_stats['original_ion_in_charge_host']}. "
                f"Rows where complex matching failed to replace ions: {match_stats['unmatched_complex']}. "
                f"Successfully created structures: {df_pairs[new_struct_col_name].notna().sum()}.")
