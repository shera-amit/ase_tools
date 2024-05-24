import numpy as np
from collections import defaultdict


def tag_atoms_by_unique_z(atoms, tolerance):
    """
    Tags atoms based on their unique Z-axis positions with a specified tolerance.

    Parameters:
    - atoms (ase.Atoms): The atomic structure.
    - tolerance (float): The tolerance for Z-axis position to consider an atom within the same layer.

    Returns:
    - tuple: A tuple containing the modified copy of the atoms object and a dictionary with unique Z positions (within tolerance) as keys and lists of atom indices as values.
    """
    # Copy the atoms object to avoid modifying the original
    atoms_copy = atoms.copy()

    # Get the Z positions of all atoms
    z_positions = atoms_copy.positions[:, 2]

    # Sort atoms by their Z positions
    sorted_indices = np.argsort(z_positions)
    sorted_z_positions = z_positions[sorted_indices]

    # Initialize a dictionary to hold atoms in each unique Z position layer
    unique_layers = defaultdict(list)

    # Initialize tags
    layer_tag = 0

    # Tag atoms based on their unique Z position within tolerance
    current_layer_z = sorted_z_positions[0]
    current_layer_indices = [sorted_indices[0]]
    atoms_copy[sorted_indices[0]].tag = layer_tag

    for i in range(1, len(sorted_z_positions)):
        if abs(sorted_z_positions[i] - current_layer_z) <= tolerance:
            current_layer_indices.append(sorted_indices[i])
            atoms_copy[sorted_indices[i]].tag = layer_tag
        else:
            unique_layers[current_layer_z].extend(current_layer_indices)
            layer_tag += 1
            current_layer_z = sorted_z_positions[i]
            current_layer_indices = [sorted_indices[i]]
            atoms_copy[sorted_indices[i]].tag = layer_tag

    # Add the last layer
    unique_layers[current_layer_z].extend(current_layer_indices)

    # Ensure the last set of indices are tagged
    for index in current_layer_indices:
        atoms_copy[index].tag = layer_tag

    return atoms_copy, unique_layers
