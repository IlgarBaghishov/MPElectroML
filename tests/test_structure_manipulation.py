# tests/test_structure_manipulation.py
import pytest
from mpelectroml.structure_manipulation import generate_multiples


# --- Tests for generate_multiples ---
def test_generate_multiples_positive_integer():
    """Test generate_multiples with various positive integers."""
    assert sorted(generate_multiples(1)) == sorted([[1, 1, 1]])

    # For n=6, expected are permutations like (1,1,6), (1,2,3), etc.
    # The function returns ordered triplets.
    multiples_of_6 = generate_multiples(6)
    expected_6 = sorted([
        [1, 1, 6], [1, 2, 3], [1, 3, 2], [1, 6, 1],
        [2, 1, 3], [2, 3, 1],
        [3, 1, 2], [3, 2, 1],
        [6, 1, 1]
    ])
    assert sorted(multiples_of_6) == expected_6
    for m in multiples_of_6:
        assert len(m) == 3
        assert m[0] * m[1] * m[2] == 6
        assert all(isinstance(x, int) and x > 0 for x in m)


def test_generate_multiples_prime_number():
    """Test generate_multiples with a prime number."""
    multiples_of_7 = generate_multiples(7)
    expected_7 = sorted([[1, 1, 7], [1, 7, 1], [7, 1, 1]])
    assert sorted(multiples_of_7) == expected_7


def test_generate_multiples_larger_number():
    """Test generate_multiples with a larger composite number."""
    multiples_of_12 = generate_multiples(12)
    # Check if product is correct for all generated triplets
    for m in multiples_of_12:
        assert m[0] * m[1] * m[2] == 12
    # Verify a known triplet is present (order matters for this function's output)
    assert [2, 2, 3] in multiples_of_12
    assert [1, 3, 4] in multiples_of_12
    # Check count if known, e.g. for 12: (1,1,12), (1,2,6), (1,3,4), (1,4,3), (1,6,2), (1,12,1) -> 6
    # (2,1,6), (2,2,3), (2,3,2), (2,6,1) -> 4
    # (3,1,4), (3,2,2), (3,4,1) -> 3
    # (4,1,3), (4,3,1) -> 2
    # (6,1,2), (6,2,1) -> 2
    # (12,1,1) -> 1. Total = 6+4+3+2+2+1 = 18.
    assert len(multiples_of_12) == 18


def test_generate_multiples_invalid_input():
    """Test generate_multiples with invalid inputs (zero, negative, float)."""
    with pytest.raises(ValueError, match="Input n must be a positive integer"):
        generate_multiples(0)
    with pytest.raises(ValueError, match="Input n must be a positive integer"):
        generate_multiples(-5)
    with pytest.raises(ValueError, match="Input n must be a positive integer"):
        generate_multiples(5.5)

# --- Placeholder for tests for create_new_working_ion_discharge_structures ---
# These tests would be more involved, requiring mock DataFrames and Pymatgen Structure objects.
# Example structure:
# def test_create_new_working_ion_simple_replacement():
#     # Setup mock DataFrame with simple LiCoO2 type structures
#     # Call create_new_working_ion_discharge_structures to replace Li with Na
#     # Assert that the new 'Na_discharge_structure' column is created correctly
#     # Assert that the formula in 'Na_discharge_formula' is correct (e.g., NaCoO2)
#     pass

# def test_create_new_working_ion_supercell_logic():
#     # Setup mock DataFrame where charge and discharge hosts require supercell matching
#     # Call create_new_working_ion_discharge_structures
#     # Assert that the supercell logic was triggered (e.g., check logs or resulting structure size)
#     # Assert that the ion replacement happened correctly in the scaled structure
#     pass

# def test_create_new_working_ion_no_original_ion():
#     # Setup mock DataFrame where discharge structure does not contain the original_working_ion
#     # Call create_new_working_ion_discharge_structures
#     # Assert that the new structure column remains None or as expected
#     pass
