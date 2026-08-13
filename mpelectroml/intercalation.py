"""
Interstitial intercalation pipeline.

Builds intercalation data for arbitrary host materials (not just Materials Project
insertion-electrode pairs): a minimal stable host is constructed, working ions are
inserted one at a time into Voronoi-discovered voids until insertion stops being
favourable, and the inserted ions are then swapped for a second working ion.

The per-material physics lives in the functions below; `add_intercalation_data_to_df`
drives them over a DataFrame and checkpoints to HDF5, mirroring
`calculations.add_energy_forces_to_df`.

Energies (not voltages) are stored. Voltages are derived downstream by
`datasets.compute_voltages`, so that every dataset shares one voltage convention.
"""
import json
import logging
import os
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from pymatgen.core import Structure

from mpelectroml.calculations import relax_structure
from mpelectroml.structure_manipulation import (
    MIN_ION_HOST_DISTANCE,
    cell_growth_exceeded,
    framework_bonds_changed,
    get_interstitial_sites,
    get_inserted_ion_indices,
)
from mpelectroml.utils import HDF5_KEY_INTERCALATION

logger = logging.getLogger(__name__)

# Statuses process_structure can return. A row carrying one of these has been fully
# processed, so a resumed run skips it. Every code path ends in one of these, which is
# what makes resuming safe: a row is either untouched ("") or finished.
COMPLETED_STATUSES = frozenset({
    "sodiated", "sodiate_error", "sodiate_not_applicable", "intercalate_error",
    "mp_relax_error", "no_voronoi_sites", "no_host", "first_ion_checked", "error",
    "processing_error",
})


@dataclass
class IntercalationSettings:
    """
    Configuration for the intercalation pipeline.

    Defaults reproduce the settings used for the full-MP interstitial dataset. Model and
    relaxation choices are deliberately exposed here rather than fixed in the library, so
    that a run can pick its own MLIP without touching library defaults used elsewhere.

    Attributes:
        working_ion: Ion inserted during intercalation (e.g. "Li").
        new_working_ion: Ion the inserted ions are swapped for afterwards (e.g. "Na").
        bulk_energies: Per-atom bulk reference energy (eV/atom) for each ion. These set
            the insertion criterion; they are not SHE-referenced (see datasets.compute_voltages).
        min_host_distance: Minimum ion-host distance (Angstrom) per ion.
        merge_distance: Candidate sites closer than this are merged.
        symprec: Symmetry tolerance for deduplicating candidate sites.
        dedup_distance: Two sites within this distance count as identical.
        max_cell_growth: Reject a structure if any cell vector grows by more than this fraction.
        first_ion_energy_cutoff: For hosts that do not already contain the working ion,
            the first insertion must be at least this favourable (eV) to be pursued.
        fmax, steps, optimizer, optimizer_kwargs: Relaxation settings.
        model_name, device, task_name, cache_dir: MLIP settings passed to assign_calculator.
    """
    working_ion: str = "Li"
    new_working_ion: str = "Na"
    bulk_energies: dict = field(default_factory=lambda: {"Li": -1.9032, "Na": -1.3093})
    min_host_distance: dict = field(default_factory=lambda: dict(MIN_ION_HOST_DISTANCE))
    merge_distance: float = 0.25
    symprec: float = 0.1
    dedup_distance: float = 0.1
    max_cell_growth: float = 0.15
    first_ion_energy_cutoff: float = -1.5
    fmax: float = 0.02
    steps: int = 500
    optimizer: str = "fire"
    optimizer_kwargs: dict = field(default_factory=lambda: {
        "dt": 0.05, "maxstep": 0.1, "dtmax": 0.2, "downhill_check": False})
    model_name: str = "uma-s-1p2"
    device: str = "cuda"
    task_name: str = "omat"
    cache_dir: str | None = None


def _relax(structure, settings, reference_structure=None):
    """Relaxes `structure` with this run's settings; see calculations.relax_structure."""
    return relax_structure(
        structure,
        fmax=settings.fmax,
        steps=settings.steps,
        optimizer=settings.optimizer,
        optimizer_kwargs=settings.optimizer_kwargs,
        reference_structure=reference_structure,
        working_ion=settings.working_ion,
        model_name=settings.model_name,
        device=settings.device,
        task_name=settings.task_name,
        cache_dir=settings.cache_dir,
    )


def _candidate_sites(structure, settings):
    """Symmetry-unique interstitial sites for the working ion, per this run's settings."""
    return get_interstitial_sites(
        structure,
        settings.working_ion,
        min_host_distance=settings.min_host_distance.get(settings.working_ion),
        merge_distance=settings.merge_distance,
        symprec=settings.symprec,
        dedup_distance=settings.dedup_distance,
    )


def screen_first_ion(structure: Structure, settings: IntercalationSettings) -> tuple:
    """
    For a host that does not already contain the working ion, finds the most favourable
    single-ion insertion energy. Used to decide whether the host is worth pursuing.

    Args:
        structure (Structure): Candidate host, as retrieved from MP.
        settings (IntercalationSettings): Run settings.

    Returns:
        tuple: (status, best_dE). Status is "first_ion_checked", "no_voronoi_sites", or
               "mp_relax_error". best_dE is float("inf") if no candidate relaxed cleanly.
    """
    ion = settings.working_ion
    bulk_energy = settings.bulk_energies[ion]

    converged, base_energy, relaxed = _relax(structure, settings)
    if not converged:
        return "mp_relax_error", 0

    candidate_sites = _candidate_sites(relaxed, settings)
    best_dE = float("inf")
    for frac in candidate_sites:
        candidate = relaxed.copy()
        candidate.append(ion, frac)

        candidate_converged, candidate_energy, _ = _relax(candidate, settings)
        if not candidate_converged:
            continue

        dE = candidate_energy - base_energy - bulk_energy
        if dE < best_dE:
            best_dE = dE

    status = "first_ion_checked" if candidate_sites else "no_voronoi_sites"
    return status, best_dE


def build_minimal_stable_host(structure: Structure, settings: IntercalationSettings) -> tuple:
    """
    Finds the smallest working-ion content at which the framework is still stable.

    The bare framework (all working ions removed) is relaxed first. If it is unstable
    (unconverged, framework bonds changed, or the cell grew too much), ions are added back
    one at a time at whichever original site gives the lowest energy, until the structure
    is stable or every ion has been restored.

    Args:
        structure (Structure): The full MP structure (contains the working ion).
        settings (IntercalationSettings): Run settings.

    Returns:
        tuple: (host_energy_per_atom, overall_dE, data, N, host_json, status), where `data`
               is the trajectory seeded at the full MP composition, `N` is the number of
               ions between the host and that composition, and status is
               "host_structure_found", "no_host", or "error".
    """
    ion = settings.working_ion
    bulk_energy = settings.bulk_energies[ion]

    # Original working-ion positions; these are the candidate re-insertion sites.
    ion_sites = [site.frac_coords for site in structure if site.specie.symbol == ion]
    num_ions = len(ion_sites)

    # Bare framework = original structure with every working ion removed.
    framework = structure.copy()
    framework.remove_species([ion])
    if len(framework) == 0:
        # Nothing left once the ion is stripped (e.g. elemental Li).
        return float("inf"), 0, [], 0, None, "no_host"

    def is_stable(candidate, converged):
        return (converged
                and not framework_bonds_changed(framework, candidate, ion)
                and not cell_growth_exceeded(candidate, framework, settings.max_cell_growth))

    converged, host_energy, host = _relax(framework.copy(), settings)
    used = set()   # indices into ion_sites already re-added to the host
    while not is_stable(host, converged) and len(used) < num_ions:
        best_energy, best_structure, best_index = float("inf"), None, None
        for index, frac in enumerate(ion_sites):
            if index in used:   # this original site already holds a re-added ion
                continue
            candidate = host.copy()
            candidate.append(ion, frac)

            candidate_converged, candidate_energy, relaxed = _relax(candidate, settings)
            if candidate_converged and candidate_energy < best_energy:
                best_energy, best_structure, best_index = candidate_energy, relaxed, index

        if best_structure is None:
            # No remaining site relaxes cleanly; accept the current host as-is.
            break
        host, host_energy, converged = best_structure, best_energy, True
        used.add(best_index)

    n_added = len(host) - len(framework)

    # Energy from the host up to the original full composition.
    converged, full_energy, full_relaxed = _relax(structure, settings)
    if not converged:
        return float("inf"), 0, [], 0, None, "error"

    # N = ions still missing between the stable host and the full structure.
    N = num_ions - n_added
    if N <= 0:
        N, overall_dE = 0, 0
    else:
        overall_dE = full_energy - host_energy - N * bulk_energy

    return (host_energy / len(host), overall_dE,
            [(N, full_relaxed.to(fmt="json"))], N, host.to(fmt="json"), "host_structure_found")


def intercalate_step(data: list, initial_structure_json: str,
                     settings: IntercalationSettings) -> tuple:
    """
    Inserts one more working ion at the best available void and appends the result.

    Args:
        data (list): Trajectory of (N, structure_json) pairs; the last entry is extended.
        initial_structure_json (str): The original MP structure, used as the framework
            topology and cell-size reference.
        settings (IntercalationSettings): Run settings.

    Returns:
        tuple: (data, N, status). Status is "intercalating" if a composition was accepted,
               "intercalated" if the structure is full, or "intercalate_error".
    """
    ion = settings.working_ion
    bulk_energy = settings.bulk_energies[ion]
    initial_structure = Structure.from_str(initial_structure_json, fmt="json")

    # Continue from the last computed composition; the new target has one more ion.
    previous_N, previous_json = data[-1]
    N = previous_N + 1

    # Re-relax the previous structure to get its reference energy.
    converged, previous_energy, previous_structure = _relax(
        Structure.from_str(previous_json, fmt="json"), settings)
    if not converged:
        return data, data[-1][0], "intercalate_error"

    candidate_sites = _candidate_sites(previous_structure, settings)
    if not candidate_sites:
        return data, data[-1][0], "intercalated"

    # Insert an ion at each candidate site and keep the lowest-dE result.
    best_dE, best_structure = float("inf"), None
    for frac in candidate_sites:
        candidate = previous_structure.copy()
        candidate.append(ion, frac)

        candidate_converged, candidate_energy, relaxed = _relax(
            candidate, settings, reference_structure=initial_structure)
        if not candidate_converged:
            continue

        dE = candidate_energy - previous_energy - bulk_energy  # cost of inserting this one ion
        if dE < best_dE:
            best_dE, best_structure = dE, relaxed

    # Stop if no candidate worked, the cell grew too much, or insertion is unfavourable.
    if (best_structure is None
            or cell_growth_exceeded(best_structure, initial_structure, settings.max_cell_growth)
            or best_dE > 0):
        return data, data[-1][0], "intercalated"

    data = data + [(N, best_structure.to(fmt="json"))]
    return data, N, "intercalating"


def get_ion_energy(data: list, host_json: str, settings: IntercalationSettings):
    """
    Energies of the host and the fully intercalated structure.

    Returns:
        tuple | None: (host_energy_per_atom, ion_energy_per_atom, dE), or None if either
                      relaxation failed.
    """
    N = data[-1][0]
    host = Structure.from_str(host_json, fmt="json")
    full = Structure.from_str(data[-1][1], fmt="json")

    host_converged, host_energy, _ = _relax(host, settings)
    full_converged, full_energy, _ = _relax(full, settings)
    if not host_converged or not full_converged:
        return None

    return (host_energy / len(host),
            full_energy / len(full),
            full_energy - host_energy - N * settings.bulk_energies[settings.working_ion])


def swap_working_ion(data: list, host_json: str, settings: IntercalationSettings) -> tuple:
    """
    Swaps the *inserted* ions for `settings.new_working_ion` and recomputes the energy.

    Ions that were already present in the host are left untouched.

    Returns:
        tuple: (dE, energy_per_atom, structure_json, status). Status is "sodiated",
               "sodiate_not_applicable" (nothing was inserted), or "sodiate_error".
    """
    ion, new_ion = settings.working_ion, settings.new_working_ion
    full = Structure.from_str(data[-1][1], fmt="json")
    host = Structure.from_str(host_json, fmt="json")

    inserted_indices = get_inserted_ion_indices(full, host, ion)
    if len(inserted_indices) == 0:
        return float("inf"), None, None, "sodiate_not_applicable"

    # Build the new-ion structure by swapping only the inserted sites.
    swapped = full.copy()
    for index in inserted_indices:
        swapped.replace(index, new_ion)

    host_converged, host_energy, _ = _relax(host, settings)
    full_converged, swapped_energy, relaxed = _relax(swapped, settings)

    # If the new-ion structure itself failed, there is no result to report.
    if not full_converged:
        return float("inf"), None, None, "sodiate_error"

    energy_per_atom = swapped_energy / len(swapped)
    structure_json = relaxed.to(fmt="json")

    # The host reference is required for the energy difference.
    if not host_converged:
        return float("inf"), energy_per_atom, structure_json, "sodiate_error"

    dE = swapped_energy - host_energy - len(inserted_indices) * settings.bulk_energies[new_ion]
    return dE, energy_per_atom, structure_json, "sodiated"


def process_structure(mp_structure_json: str, settings: IntercalationSettings) -> dict:
    """
    Full per-material pipeline: build host, intercalate to saturation, swap the ion.

    Args:
        mp_structure_json (str): The MP structure as a JSON string.
        settings (IntercalationSettings): Run settings.

    Returns:
        dict: Column name -> value for this material. Structures are JSON strings.
    """
    ion, new_ion = settings.working_ion, settings.new_working_ion
    result = {
        "status": "unprocessed",
        "N": -1,
        "data": "[]",
        "host_structure": "",
        "host_energy_per_atom": np.nan,
        f"{ion}_structure": "",
        f"{ion}_energy_per_atom": np.nan,
        f"{ion}_dE": np.nan,
        f"{new_ion}_structure": "",
        f"{new_ion}_energy_per_atom": np.nan,
        f"{new_ion}_dE": np.nan,
    }

    structure = Structure.from_str(mp_structure_json, fmt="json")
    has_ion = ion in {element.symbol for element in structure.composition.elements}
    data = [(0, mp_structure_json)]
    host_json = ""

    if has_ion:
        # The host must be carved out of the structure before anything can be inserted.
        host_energy_per_atom, _, host_data, N, host_json, status = build_minimal_stable_host(
            structure, settings)
        if status == "host_structure_found":
            data = host_data
            result.update({"host_energy_per_atom": host_energy_per_atom, "N": N,
                           "host_structure": host_json,
                           "data": json.dumps([[n, struct] for n, struct in data])})
        result["status"] = status
    else:
        # No native working ion: only pursue hosts where the first insertion is favourable.
        status, first_ion_dE = screen_first_ion(structure, settings)
        if status == "first_ion_checked" and first_ion_dE < settings.first_ion_energy_cutoff:
            converged, host_energy, host = _relax(structure, settings)
            host_json = host.to(fmt="json")
            result.update({"host_energy_per_atom": host_energy / len(host),
                           "N": 0, "host_structure": host_json})
            status = "host_structure_found"
        result["status"] = status

    if result["status"] == "host_structure_found":
        status = "intercalating"
        while status == "intercalating":
            data, N, status = intercalate_step(data, mp_structure_json, settings)
            # The trajectory is only recorded once intercalation actually runs; materials
            # that never reach a usable host keep an empty trajectory.
            result.update({"N": N, "status": status,
                           "data": json.dumps([[n, struct] for n, struct in data])})

    if result["status"] == "intercalated":
        energies = get_ion_energy(data, host_json, settings)
        if energies is None:
            result["status"] = "intercalate_error"
            return result

        host_energy_per_atom, ion_energy_per_atom, ion_dE = energies
        result.update({"host_energy_per_atom": host_energy_per_atom,
                       f"{ion}_energy_per_atom": ion_energy_per_atom,
                       f"{ion}_dE": ion_dE,
                       f"{ion}_structure": data[-1][1]})

        new_dE, new_energy_per_atom, new_structure_json, status = swap_working_ion(
            data, host_json, settings)
        if status == "sodiated":
            result.update({f"{new_ion}_dE": new_dE,
                           f"{new_ion}_energy_per_atom": new_energy_per_atom,
                           f"{new_ion}_structure": new_structure_json})
        result["status"] = status

    return result


def _load_previous_results(df, output_path, structure_column):
    """
    Returns a previously written frame to continue from, or None.

    Only adopts the old frame when it is unambiguously the same work: same row count and
    identical structures in the same order. Anything else is refused rather than risking
    results being written against the wrong rows.
    """
    if not os.path.exists(output_path):
        return None
    try:
        previous = pd.read_hdf(output_path)
    except Exception as e:
        logger.warning(f"Could not read {output_path} to resume ({e}); starting fresh.")
        return None

    if len(previous) != len(df):
        logger.warning(f"Refusing to resume: {output_path} has {len(previous)} rows, "
                       f"input has {len(df)}. Starting fresh.")
        return None
    if structure_column not in previous.columns:
        logger.warning(f"Refusing to resume: {output_path} lacks '{structure_column}'. Starting fresh.")
        return None

    old = previous[structure_column].reset_index(drop=True)
    new = df[structure_column].reset_index(drop=True)
    if not old.equals(new):
        logger.warning(f"Refusing to resume: structures in {output_path} do not match the "
                       "input frame. Starting fresh.")
        return None

    return previous.reset_index(drop=True)


def add_intercalation_data_to_df(
    df: pd.DataFrame,
    settings: IntercalationSettings,
    file_dirpath: str,
    structure_column: str = "mp_structure",
    idx_init: int = 0,
    idx_final: int = -1,
    checkpoint_every: int = 100,
    output_filename: str | None = None,
    resume: bool = True,
    redo_statuses: set | None = None
) -> pd.DataFrame:
    """
    Runs the intercalation pipeline over rows `idx_init` to `idx_final` of `df`.

    Modifies `df` in place and checkpoints to HDF5 every `checkpoint_every` rows, matching
    the resume/shard behaviour of `calculations.add_energy_forces_to_df`. To use several
    GPUs, run one process per shard with disjoint index ranges and concatenate the outputs.

    Args:
        df (pd.DataFrame): Must contain `structure_column` holding JSON structure strings.
        settings (IntercalationSettings): Run settings.
        file_dirpath (str): Directory for the HDF5 output.
        structure_column (str): Column holding the MP structure JSON.
        idx_init (int): First row index to process.
        idx_final (int): One past the last row index; -1 means to the end.
        checkpoint_every (int): Rows between HDF5 checkpoints.
        output_filename (str | None): Override the default output file name.
        resume (bool): If an output file for the same structures exists, continue from it
            and skip rows already carrying a completed status.
        redo_statuses (set | None): Statuses to recompute rather than skip when resuming,
            e.g. {"processing_error"} to retry rows lost to a transient failure.

    Returns:
        pd.DataFrame: The DataFrame with intercalation columns filled in. When resuming,
        this is the reloaded frame, so callers should use the return value.
    """
    os.makedirs(file_dirpath, exist_ok=True)
    if output_filename is None:
        output_filename = f"{settings.working_ion}_{settings.new_working_ion}_intercalation_data.h5"
    output_path = os.path.join(file_dirpath, output_filename)

    if structure_column not in df.columns:
        logger.error(f"Structure column '{structure_column}' not found in DataFrame.")
        return df

    if resume:
        previous = _load_previous_results(df, output_path, structure_column)
        if previous is not None:
            df = previous
            logger.info(f"Resuming from {output_path}.")

    ion, new_ion = settings.working_ion, settings.new_working_ion
    text_columns = ["status", "data", "host_structure", f"{ion}_structure", f"{new_ion}_structure"]
    float_columns = ["host_energy_per_atom", f"{ion}_energy_per_atom", f"{ion}_dE",
                     f"{new_ion}_energy_per_atom", f"{new_ion}_dE"]

    for column in text_columns:
        if column not in df.columns:
            df[column] = [""] * len(df)
            df[column] = df[column].astype(object)
    for column in float_columns:
        if column not in df.columns:
            df[column] = np.nan
    if "N" not in df.columns:
        df["N"] = -1

    if idx_final == -1:
        idx_final = df.shape[0]
    idx_init = max(0, idx_init)
    idx_final = min(df.shape[0], idx_final)

    skip_statuses = COMPLETED_STATUSES - set(redo_statuses or ())

    logger.info(f"Starting intercalation ({ion} -> {new_ion}) with model '{settings.model_name}'.")
    logger.info(f"Processing DataFrame rows from index {idx_init} to {idx_final - 1}.")
    already_done = sum(1 for i in range(idx_init, idx_final) if df.at[i, "status"] in skip_statuses)
    if already_done:
        logger.info(f"Skipping {already_done} row(s) already completed.")

    for i in range(idx_init, idx_final):
        if df.at[i, "status"] in skip_statuses:
            continue
        logger.info(f"Processing row {i}...")
        try:
            result = process_structure(df.at[i, structure_column], settings)
        except Exception as e:
            logger.error(f"Row {i} failed: {type(e).__name__}: {e}")
            result = {"status": "processing_error"}

        for column, value in result.items():
            df.at[i, column] = value

        if (i + 1) % checkpoint_every == 0:
            logger.info(f"Processed up to row {i}, saving intermediate results to {output_path}...")
            try:
                df.to_hdf(output_path, key=HDF5_KEY_INTERCALATION, mode="w")
            except Exception as e:
                logger.error(f"Error saving intermediate DataFrame to HDF5: {e}")

    logger.info(f"All specified rows ({idx_init} to {idx_final - 1}) processed.")
    try:
        df.to_hdf(output_path, key=HDF5_KEY_INTERCALATION, mode="w")
        logger.info(f"Final DataFrame saved to {output_path} (key: {HDF5_KEY_INTERCALATION}).")
    except Exception as e:
        logger.error(f"Error saving final DataFrame to HDF5: {e}")

    return df
