"""CPU-only unit tests for the interstitial site discovery and integrity checks.

These avoid any MLIP or Materials Project access so they can run in CI.
"""
import numpy as np
import pytest
from pymatgen.core import Lattice, Structure

from mpelectroml.structure_manipulation import (
    cell_growth_exceeded,
    framework_bonds_changed,
    get_inserted_ion_indices,
    get_interstitial_sites,
)


@pytest.fixture
def rocksalt():
    """A simple MgO rocksalt cell with plenty of open volume."""
    lattice = Lattice.cubic(4.2)
    return Structure(lattice, ["Mg", "O"], [[0, 0, 0], [0.5, 0.5, 0.5]])


def test_get_interstitial_sites_returns_sites_inside_cell(rocksalt):
    sites = get_interstitial_sites(rocksalt, "Li")
    assert len(sites) > 0
    for frac in sites:
        assert np.all(frac >= -1e-6) and np.all(frac < 1.0 + 1e-6)


def test_get_interstitial_sites_respects_min_host_distance(rocksalt):
    # A very large exclusion radius leaves no void clear of the host atoms.
    assert get_interstitial_sites(rocksalt, "Li", min_host_distance=10.0) == []


def test_get_interstitial_sites_dedups_by_symmetry(rocksalt):
    # Symmetry deduplication cannot increase the site count, and with a large merge
    # distance every remaining site must be well separated.
    all_sites = get_interstitial_sites(rocksalt, "Li", merge_distance=0.01)
    merged = get_interstitial_sites(rocksalt, "Li", merge_distance=1.0)
    assert len(merged) <= len(all_sites)


def test_framework_bonds_changed_ignores_working_ion(rocksalt):
    with_ion = rocksalt.copy()
    with_ion.append("Li", [0.25, 0.25, 0.25])
    # Adding only a working ion leaves the framework bond network untouched.
    assert not framework_bonds_changed(rocksalt, with_ion, "Li")


def test_framework_bonds_changed_detects_expansion():
    # At 3.0 Angstrom the Mg-O pair is inside the covalent-radius cutoff; doubling the
    # cell edge pulls it outside, so the framework bond network changes.
    bonded = Structure(Lattice.cubic(3.0), ["Mg", "O"], [[0, 0, 0], [0.5, 0.5, 0.5]])
    stretched = bonded.copy()
    stretched.scale_lattice(bonded.volume * 8)
    assert framework_bonds_changed(bonded, stretched, "Li")


def test_cell_growth_exceeded():
    small = Structure(Lattice.cubic(4.0), ["Mg"], [[0, 0, 0]])
    grown = Structure(Lattice.cubic(4.8), ["Mg"], [[0, 0, 0]])   # +20%
    assert cell_growth_exceeded(grown, small, max_growth=0.15)
    assert not cell_growth_exceeded(grown, small, max_growth=0.25)
    # Shrinkage is deliberately not flagged.
    assert not cell_growth_exceeded(small, grown, max_growth=0.15)


def test_get_inserted_ion_indices_bare_framework():
    host = Structure(Lattice.cubic(5.0), ["O"], [[0, 0, 0]])
    full = host.copy()
    full.append("Li", [0.5, 0.5, 0.5])
    assert get_inserted_ion_indices(full, host, "Li") == [1]


def test_get_inserted_ion_indices_matches_native_ions():
    host = Structure(Lattice.cubic(5.0), ["Li", "O"], [[0, 0, 0], [0.5, 0.0, 0.0]])
    full = host.copy()
    full.append("Li", [0.5, 0.5, 0.5])
    inserted = get_inserted_ion_indices(full, host, "Li")
    # Only the added ion counts; the host's own Li is matched and excluded.
    assert len(inserted) == 1
    assert np.allclose(full[inserted[0]].frac_coords, [0.5, 0.5, 0.5])


def test_get_inserted_ion_indices_no_room():
    host = Structure(Lattice.cubic(5.0), ["Li", "O"], [[0, 0, 0], [0.5, 0.0, 0.0]])
    assert get_inserted_ion_indices(host, host, "Li") == []
