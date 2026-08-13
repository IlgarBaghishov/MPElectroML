# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this project is

MPElectroML (Materials Project Electrodes with Machine-Learned Interatomic Potentials) fetches insertion-electrode data from the Materials Project, substitutes the working ion in known host structures (e.g. Li → Na, K, Mg, Ca), and computes energies/forces for the original and substituted structures with machine-learned interatomic potentials. Output is a pandas DataFrame persisted as HDF5. The intended use is computational screening / data generation for novel battery electrode materials.

## Commands

Requires an active Materials Project API key for any step that hits MP:
```bash
export MP_API_KEY="YOUR_KEY"
```

- Install (editable, with dev/test extras): `pip install -e .[dev]`  (or `.[test]` for just test deps)
- Lint (matches CI): `flake8 mpelectroml/ tests/ examples/ --count --max-complexity=25 --max-line-length=119 --statistics`
- Run the test suite: `pytest tests/ -v --cov=mpelectroml --cov-report=term-missing`
- Run a single test: `pytest tests/test_structure_manipulation.py::test_generate_multiples_prime_number -v`

CI (`.github/workflows/ci.yml`) runs flake8 + pytest on Python 3.10/3.11/3.12. Tests must not make real MP API calls — CI provides only a dummy `MP_API_KEY`. The only covered function today is `generate_multiples`; the MP-/MLIP-dependent code paths are untested.

## Architecture

The library is the `mpelectroml/` package; `__init__.py` re-exports the public functions. The pipeline is four sequential stages, each a function that **mutates a shared `df_pairs` DataFrame in place** and (for calculations) writes HDF5 checkpoints:

1. `data_retrieval.get_electrode_pairs(working_ion, api_key, fields)` — queries `mpr.materials.insertion_electrodes.search`, returns a DataFrame of `charge_id`/`discharge_id`/`working_ion`.
2. `data_retrieval.get_structures_from_electrode_pair_ids(df_pairs, api_key, fields)` — batches `mpr.materials.summary.search` (BATCH_SIZE=5000) to add `charge_*`/`discharge_*` columns (structure, energy_per_atom, formula). Column names are derived programmatically from the requested `fields` list, so changing `MP_SUMMARY_FIELDS` changes the column set — `structure` and `energy_per_atom` are assumed downstream.
3. `structure_manipulation.create_new_working_ion_discharge_structures(df_pairs, original_working_ion, new_working_ion)` — builds `{new_ion}_discharge_structure`/`_formula` columns by replacing the working ion. The non-trivial part: when the charge and discharge host frameworks (working ion removed) differ in size, it scales the smaller one with `make_supercell` using factor options from `generate_multiples`, then uses a loose `StructureMatcher` (ltol=0.6, stol=0.8, angle_tol=20, allow_subset=True) to map sites and substitute only the ion sites that don't map to the charge host. Passing `original_working_ion=''` makes it read the per-row `working_ion` column instead.
4. `calculations.add_energy_forces_to_df(df_pairs, original_working_ion, ion_type_to_process, file_dirpath, model_name, idx_init, idx_final)` — for one structure type (`"charge"`, `"discharge"`, or an ion name like `"Na"` → `{ion}_discharge_structure`), computes both initial (unrelaxed) and relaxed energy/forces and adds `{prefix}_init_*` / `{prefix}_relaxed_*` columns. Saves to `{original_working_ion}_electrode_data_with_energies.h5` every 100 rows and at the end. `idx_init`/`idx_final` allow resuming or sharding a long run.

`calculate_energy_and_forces_from_Structure(structure, model_name, relax, ...)` is the single-structure core supporting two backends selected by `model_name`:
- `"uma"` (default): FAIRChem `pretrained_mlip` predictor → ASE `Atoms` with `FAIRChemCalculator`, relaxed via `LBFGS` over a `FrechetCellFilter` (cell + positions). Energy is normalized to per-atom.
- `"chgnet"`: CHGNet `predict_structure` / `StructOptimizer.relax`.

Both backends lazily initialize their model into a module-level singleton (`_FAIRCHEM_PREDICTOR`, `_CHGNET_PREDICTOR`, `_CHGNET_RELAXER`) so the model is loaded once per process.

### Second pipeline: full-MP interstitial intercalation

`intercalation.py` implements a second, independent pipeline that does **not** rely on MP
insertion-electrode pairs. It starts from arbitrary hosts, discovers its own insertion
sites, and fills them iteratively:

1. `data_retrieval.get_materials_summary(api_key, fields, **search_kwargs)` - generic
   wrapper over `mpr.materials.summary.search`; selection filters (e.g. `theoretical=False`,
   `energy_above_hull=(0, 0.1)`) are passed by the caller, not baked into the library.
   Structures are returned as **JSON strings** (not pickled Structure objects) so the
   resulting HDF5 stays readable across numpy/pymatgen versions.
2. `structure_manipulation.get_interstitial_sites(structure, working_ion, ...)` - candidate
   voids from Voronoi vertices of a 3x3x3 supercell, filtered by host distance, merged when
   near-coincident, then deduplicated by space-group orbit.
3. `intercalation.build_minimal_stable_host` - strips the working ion, relaxes, and adds
   ions back one at a time until the framework is stable (converged, bond network unchanged
   per `framework_bonds_changed`, cell not grown past `max_cell_growth`).
4. `intercalation.intercalate_step` - inserts one ion at the best void, repeating until
   insertion becomes unfavourable (dE > 0), the cell grows too much, or no voids remain.
5. `intercalation.swap_working_ion` - swaps only the *inserted* ions (identified by
   `get_inserted_ion_indices`) for the second ion and recomputes the energy.

`add_intercalation_data_to_df` drives these over a DataFrame with the same
checkpoint/shard contract as `add_energy_forces_to_df` (`idx_init`/`idx_final`, periodic
`to_hdf`). There is no multiprocessing pool: parallelism comes from running one process
per shard, as elsewhere in the repo.

`calculations.relax_structure(structure, ..., reference_structure=...)` is the shared
primitive this pipeline needs: it returns `(converged, total_energy, structure)` rather
than a per-atom energy, and can abort mid-relaxation via `BondBrokenError` if the framework
topology changes. `assign_calculator` is parameterized (`model_name`, `device`, `task_name`,
`cache_dir`, the latter defaulting to `$FAIRCHEM_CACHE_DIR`) and caches one predictor per
distinct combination, so the two pipelines can use different MLIPs in one process.

**This pipeline stores energies, not voltages.** Voltages come from
`datasets.compute_voltages`, which applies `V = -dE / (N * z) + E0(X+/X)` with the SHE
potentials in `datasets.SHE_POTENTIALS`. `datasets.split_and_export` is the single
definition of the train/test/val split used by both datasets. `normalize_dataset` maps
either pipeline onto a shared column schema and tags rows with `source`; `merge_datasets`
is a stub pending regeneration of the electrode dataset with a matching MLIP.

`utils.py` holds `get_api_key` (reads `MP_API_KEY`), `setup_logging`, and the two HDF5 keys: `HDF5_KEY_ELECTRODE_PAIRS = "data"` (pairs + structures file `{ion}_electrode_data.h5`) and `HDF5_KEY_WITH_ENERGIES = "data_with_energies"` (file with energies). Every module uses `logging.getLogger(__name__)`; functions log-and-continue rather than raise, returning `None`/empty/`[None, None, None]` on failure — check return values rather than relying on exceptions.

See `README.md` for the full column-by-column schema of the energies DataFrame.

## Hardware / environment notes

`calculations.assign_calculator` is hardcoded to `device="cuda"`, model `"uma-m-1p1"`, task `"omat"`, and `cache_dir="/scratch/08405/ilgar/.cache/farichem"` (a user-specific path). Edit these for a different machine/model/GPU — there is no config layer for them. FAIRChem performance assumes a GPU.

## examples/ directory

`examples/full_mp_interstitial/` holds the interstitial run: `get_data.py` (config +
driver) and `make_dataset.py` (HDF5 -> train/test/val CSVs). Otherwise, `examples/` is not a tidy demo folder — it is the working area for individual HPC runs (TACC Vista, SLURM). The canonical pipeline driver is `examples/run_analysis.py`: configuration is a block of module-level constants at the top (`WORKING_ION`, `NEW_WORKING_IONS`, `CALC_TYPES`, `CALC_IDX_INITS/FINALS`, `SKIP_*`/`RESUME_FROM_FILES` flags) — there is no argparse. Each subdirectory (`Li_Na_electrodes_uma_m/`, `Mg_electrodes_uma_m/`, `Na_chgnet/`, …) contains a `get_data.py` that is a copy of `run_analysis.py` with those constants edited for that experiment, plus a SLURM `run.sh` that does `python -u get_data.py`. The `*_to_test_assumptions_*`, `li_data_*`, and `combine_*`/`preprocessing` notebooks are downstream data-prep/analysis, not part of the library.

`.gitignore` deliberately excludes most run artifacts (`run.sh`, `ll_out*`, `*.h5`, `*.log`, `$SCRATCH/`) with a few explicit `!` un-ignore exceptions (e.g. `final_dataset.h5`, `combine_Li_Na copy.ipynb`). Check `.gitignore` before assuming a file under `examples/` is tracked.
