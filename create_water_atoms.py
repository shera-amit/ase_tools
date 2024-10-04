import numpy as np
from ase.build import molecule
from ase import Atoms
from ase.db import connect

# Constants
N_A = 6.02214076e23          # Avogadro's number (mol^-1)
M = 18.01528 / 1000          # Molar mass of water in kg/mol
mass_per_molecule = M / N_A  # Mass per water molecule in kg

# Parameters
N_structures = 1000           # Number of structures to generate
N_molecules = 16             # Number of water molecules per structure
mass = N_molecules * mass_per_molecule  # Total mass of water molecules in kg

# Create or connect to the ASE database
db = connect('water_structures_varying_density.db')

for i in range(N_structures):
    # Randomly select a density between 600 and 2000 kg/m^3 (0.6 to 2 g/cm^3)
    density = np.random.uniform(600, 2000)  # kg/m^3

    # Calculate the volume and cell length based on the density
    V = mass / density         # Volume in m^3
    L = V ** (1/3)             # Cell length in meters
    L_angstrom = L * 1e10      # Convert cell length to Angstroms

    # Create an empty Atoms object with periodic boundary conditions
    atoms = Atoms(cell=[L_angstrom]*3, pbc=True)

    # Generate grid positions for placing water molecules
    grid_size = int(round(N_molecules ** (1/3)))  # Size of the grid (e.g., 4 for 64 molecules)
    spacing = L_angstrom / grid_size
    offsets = np.linspace(0, L_angstrom - spacing, grid_size) + spacing / 2
    positions = np.array([[x, y, z] for x in offsets for y in offsets for z in offsets])

    # Randomize the positions to avoid systematic arrangements
    np.random.shuffle(positions)

    # Create a single water molecule template
    water = molecule('H2O')

    # Place water molecules at the generated positions
    for pos in positions:
        mol = water.copy()
        mol.translate(pos)
        atoms += mol

    # Apply heavy rattling to atom positions (standard deviation of 2.0 Å)
    displacement = np.random.normal(0, 1, size=(len(atoms), 3))
    atoms.positions += displacement
    atoms.wrap()
    # Write the structure to the database with density metadata
    db.write(atoms, density=density, description=f'Water structure with density {density:.2f} kg/m^3')

    # Optional: Print progress
    if (i + 1) % 50 == 0:
        print(f'Generated {i + 1}/{N_structures} structures')

print('Structure generation complete. All structures are saved in "water_structures.db".')
