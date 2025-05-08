import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.spatial import cKDTree # Use cKDTree for C implementation speed
from ase import Atoms
from ase.build import molecule as ase_molecule
from ase.geometry import wrap_positions # For PBC wrapping

# Constants
AVOGADRO_NUMBER = 6.02214076e23  # mol^-1
ANGSTROM_TO_CM = 1e-8  # 1 Angstrom = 1e-8 cm

def create_water_molecule():
    """Creates a single ASE Atoms object for a water molecule, centered."""
    water = ase_molecule('H2O')
    water.center(vacuum=0.0) # Center the molecule at (0,0,0) for easier rotation
    return water

def get_molecule_mass_amu(ase_atoms_molecule):
    """Calculates the mass of a molecule in AMU."""
    return np.sum(ase_atoms_molecule.get_masses())

def calculate_box_volume(cell_vectors):
    """Calculates the volume of the simulation box in Angstrom^3."""
    return np.abs(np.linalg.det(cell_vectors))

def place_molecules_in_box_kdtree(cell_vectors, density_g_cm3, min_intermolecular_distance=1.5,
                                  max_placement_attempts_per_molecule=1000):
    """
    Places water molecules randomly in a box with a given density, using KDTree
    for efficient collision detection with Periodic Boundary Conditions (PBC).

    Args:
        cell_vectors (np.ndarray): A 3x3 numpy array where rows are the cell vectors (a, b, c).
                                   Units: Angstroms.
        density_g_cm3 (float): Desired density in g/cm^3.
        min_intermolecular_distance (float): Minimum allowed distance between any two atoms
                                             of different molecules (in Angstroms).
        max_placement_attempts_per_molecule (int): Max attempts to place a single molecule.

    Returns:
        ase.Atoms or None: An ASE Atoms object containing all placed water molecules,
                           or a partially filled Atoms object if placement fails for some molecules.
                           Returns an empty Atoms object if num_target_water_molecules is 0.
    """
    if not isinstance(cell_vectors, np.ndarray) or cell_vectors.shape != (3, 3):
        raise ValueError("cell_vectors must be a 3x3 numpy array.")
    if density_g_cm3 <= 0:
        raise ValueError("Density must be positive.")
    if min_intermolecular_distance <= 0:
        raise ValueError("min_intermolecular_distance must be positive.")

    template_water = create_water_molecule()
    water_mass_amu = get_molecule_mass_amu(template_water)
    water_mass_g = water_mass_amu / AVOGADRO_NUMBER

    volume_A3 = calculate_box_volume(cell_vectors)
    volume_cm3 = volume_A3 * (ANGSTROM_TO_CM ** 3)

    total_mass_g = density_g_cm3 * volume_cm3
    num_target_water_molecules = int(round(total_mass_g / water_mass_g))

    if num_target_water_molecules == 0:
        print("Calculated number of water molecules is 0. Returning an empty Atoms object.")
        return Atoms(cell=cell_vectors, pbc=True)

    print(f"Targeting {num_target_water_molecules} water molecules.")
    print(f"Box Volume: {volume_A3:.2f} Å^3")

    all_placed_atoms = Atoms(cell=cell_vectors, pbc=True)
    num_atoms_per_molecule = len(template_water)

    pbc_offsets = []
    for i in [-1, 0, 1]:
        for j in [-1, 0, 1]:
            for k in [-1, 0, 1]:
                pbc_offsets.append(i * cell_vectors[0] +
                                   j * cell_vectors[1] +
                                   k * cell_vectors[2])
    pbc_offsets = np.array(pbc_offsets)

    placed_molecule_count = 0
    for molecule_idx in range(num_target_water_molecules):
        placed_successfully = False
        for attempt in range(max_placement_attempts_per_molecule):
            current_water = template_water.copy()

            random_rotation_matrix = R.random().as_matrix()
            current_water.positions @= random_rotation_matrix.T

            random_frac_coords = np.random.rand(3)
            com_cart_coords = random_frac_coords @ cell_vectors
            current_water.translate(com_cart_coords)

            collision_detected = False
            if len(all_placed_atoms) > 0:
                existing_positions = all_placed_atoms.get_positions()
                kdtree = cKDTree(existing_positions)

                for atom_idx_in_molecule in range(num_atoms_per_molecule):
                    new_atom_abs_pos = current_water.positions[atom_idx_in_molecule]
                    query_points_for_new_atom = new_atom_abs_pos - pbc_offsets
                    
                    # MODIFIED LINE: Removed n_jobs=-1
                    indices_of_nearby_atoms_list = kdtree.query_ball_point(
                        query_points_for_new_atom,
                        r=min_intermolecular_distance,
                        return_sorted=False # Still useful if available
                    )
                    
                    for sub_list in indices_of_nearby_atoms_list:
                        if len(sub_list) > 0:
                            collision_detected = True
                            break
                    if collision_detected:
                        break
            
            if not collision_detected:
                all_placed_atoms.extend(current_water)
                placed_successfully = True
                placed_molecule_count += 1
                break

        if not placed_successfully:
            print(f"Failed to place molecule number {molecule_idx + 1} (targeting {num_target_water_molecules}) "
                  f"after {max_placement_attempts_per_molecule} attempts for this specific molecule.")
            print("Consider reducing density, min_distance, increasing box size, or increasing max_placement_attempts.")
            print(f"Returning system with {placed_molecule_count} successfully placed molecules.")
            break 

    if len(all_placed_atoms) > 0:
        all_placed_atoms.set_positions(wrap_positions(all_placed_atoms.get_positions(),
                                                     all_placed_atoms.get_cell(),
                                                     pbc=all_placed_atoms.pbc))
    
    print(f"Successfully placed {placed_molecule_count} out of {num_target_water_molecules} targeted water molecules.")
    return all_placed_atoms

# --- Example Usage (remains the same) ---
if __name__ == "__main__":
    # Define the simulation box (e.g., a cubic box of 20x20x20 Angstroms)
    box_side_length = 30  # Angstroms
    cell = np.array([
        [box_side_length, 0, 0],
        [0, box_side_length, 0],
        [0, 0, box_side_length]
    ])

    density = 0.9  # g/cm^3 (Closer to liquid water)
    min_dist = 1.8  # Angstroms

    print("Starting molecule placement using KDTree...")
    final_system_kdtree = place_molecules_in_box_kdtree(
        cell,
        density,
        min_intermolecular_distance=min_dist,
        max_placement_attempts_per_molecule=2000 # Increased attempts for higher density
    )

    if final_system_kdtree is not None and len(final_system_kdtree) > 0:
        print(f"\nTotal atoms in the final system (KDTree): {len(final_system_kdtree)}")
        print(f"Number of water molecules: {len(final_system_kdtree) // 3}")
        print(f"Cell vectors:\n{final_system_kdtree.get_cell()}")

        try:
            from ase.io import write
            output_filename = "water_box_kdtree.xyz"
            write(output_filename, final_system_kdtree)
            print(f"System saved to {output_filename}")

            num_mols = len(final_system_kdtree) / 3
            if num_mols > 0:
                water_mass_amu_val = get_molecule_mass_amu(create_water_molecule())
                total_mass_amu = num_mols * water_mass_amu_val
                total_mass_g_calc = total_mass_amu / AVOGADRO_NUMBER
                volume_A3_calc = calculate_box_volume(final_system_kdtree.get_cell())
                volume_cm3_calc = volume_A3_calc * (ANGSTROM_TO_CM ** 3)
                if volume_cm3_calc > 1e-9: # Avoid division by zero or tiny volume
                    achieved_density = total_mass_g_calc / volume_cm3_calc
                    print(f"Achieved density (KDTree): {achieved_density:.4f} g/cm^3 (Target: {density:.4f} g/cm^3)")
                else:
                    print("Could not calculate achieved density (volume too small).")
            else:
                print("No molecules placed, cannot calculate achieved density.")

        except ImportError:
            print("ASE IO module not fully available. Cannot write file.")
        except Exception as e:
            print(f"An error occurred during file writing or density check: {e}")

    elif final_system_kdtree is not None and len(final_system_kdtree) == 0:
         print("Molecule placement (KDTree) resulted in an empty system (0 molecules placed).")
    else:
        print("Molecule placement (KDTree) failed in an unexpected way.")


    print("\n--- KDTree Example with a triclinic cell ---")
    triclinic_cell = np.array([
        [15.0, 0.0, 0.0],    # vector a
        [2.0, 18.0, 0.0],    # vector b
        [1.0, 3.0, 12.0]     # vector c
    ])
    density_triclinic = 0.75 # g/cm^3
    min_dist_triclinic = 1.7

    print("Starting KDTree molecule placement in triclinic cell...")
    final_system_triclinic_kdtree = place_molecules_in_box_kdtree(
        triclinic_cell,
        density_triclinic,
        min_intermolecular_distance=min_dist_triclinic,
        max_placement_attempts_per_molecule=2500
    )

    if final_system_triclinic_kdtree is not None and len(final_system_triclinic_kdtree) > 0:
        print(f"\nTotal atoms in the triclinic system (KDTree): {len(final_system_triclinic_kdtree)}")
        print(f"Number of water molecules: {len(final_system_triclinic_kdtree) // 3}")
        print(f"Cell vectors:\n{final_system_triclinic_kdtree.get_cell()}")
        try:
            from ase.io import write
            write("water_box_triclinic_kdtree.xyz", final_system_triclinic_kdtree)
            print("Triclinic system (KDTree) saved to water_box_triclinic_kdtree.xyz")
        except Exception as e:
            print(f"Error saving triclinic kdtree file: {e}")
    elif final_system_triclinic_kdtree is not None and len(final_system_triclinic_kdtree) == 0:
        print("Triclinic molecule placement (KDTree) resulted in an empty system.")
    else:
        print("Triclinic molecule placement (KDTree) failed in an unexpected way.")
