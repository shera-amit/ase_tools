from ase.visualize import view
from ase.io import  read
from ase.build import add_vacuum
from scm.plams import plot_molecule, from_smiles, Molecule
from scm.plams.interfaces.molecule.packmol import packmol
from ase.visualize.plot import plot_atoms
from ase.build import fcc111, bulk
import matplotlib.pyplot as plt
import importlib.metadata
import numpy as np
from scm.plams import packmol_around
from ase.db import connect
from ase import  Atoms

def make_sanwich_from_bulk(suatoms: Atoms, density :float = 1.0, box_size: int= 1):
    suatoms = suatoms.copy()
    shifted_atoms = suatoms.copy()
    a,b,c = suatoms.cell.cellpar()[:3]
    a, b, c = box_bound = suatoms.cell.cellpar()[:3]
    water = from_smiles("O")
    water_box = packmol(water, density=1.0, box_bounds=[0,0,0, a*box_size, b, c])
    water_box
    water_box.write('water_box_tmp.xyz')
    shifted_atoms = suatoms.copy()
    shifted_atoms.translate([(box_size + 1)*a, 0, 0])
    slab = suatoms.copy()
    slab.set_cell([(box_size + 2)*a, b, c])
    water_to_add = read('water_box_tmp.xyz')
    water_to_add.translate([a, 0, 0])    
    slab += shifted_atoms
    final_slab = slab + water_to_add
    final_slab.wrap()
    return final_slab
db = connect('/home/as41vomu/work_data/bto.db')
new_db = connect('combine_water.db')
for row in db.select():
    if row.id == 1 or row.id == 2: 
        atoms = row.toatoms()
        for i in np.linspace(0.6, 1.5, 200):
            try:
                suatoms = atoms.repeat((2, 3, 3))
                combine_atoms = make_sanwich_from_bulk(suatoms, density=i, box_size=1)
                combine_atoms.rattle(0.03)
                new_db.write(combine_atoms, name=f'water_{i:.2f}', data={'density': i})
            except:
                print('Error in making sandwich')
                continue
            
test_suatoms =  read('./mp-2998_relaxed.cif')
test_suatoms = test_suatoms.repeat((2,3,3))
test_combine = make_sanwich(test_suatoms, density=1.0, box_size=1)
view(test_combine)
# find min distance in atoms which are non zero
distances = test_combine.get_all_distances(mic=True)
# Create a mask to exclude zero distances (self-interactions)
non_zero_mask = distances > 1e-10  # Small threshold to handle floating point errors
# Find minimum of non-zero distances
min_dist = np.min(distances[non_zero_mask])
print(f"Minimum distance between atoms: {min_dist:.3f} Å")

def make_sanwich_from_slab(slab: Atoms, density :float = 1.0, extra_vacuum: float = 0, rattle: float = 0.05):
    suatoms = slab.copy()
    a, b, c = suatoms.cell.cellpar()[:3]
    suatoms_max_z = slab.get_positions()[:, 2].max()
    suatoms_min_z = slab.get_positions()[:, 2].min()
    
    if (suatoms_min_z > 4) and (c - suatoms_max_z > 4):
        box_1_bound = [0, 0, 0, a, b, suatoms_min_z - extra_vacuum]
        box_2_bound = [0, 0, 0, a, b, c - extra_vacuum - suatoms_max_z]
        
        water = from_smiles("O")
        water_box_1 = packmol(water, density=density, box_bounds=box_1_bound)
        water_box_2 = packmol(water, density=density, box_bounds=box_2_bound)
        
        water_box_1.write('water_box_1_tmp.xyz')
        water_box_2.write('water_box_2_tmp.xyz')
        
        water_box_1 = read('water_box_1_tmp.xyz')
        water_box_2 = read('water_box_2_tmp.xyz')
        
        water_box_1.rattle(rattle)
        water_box_2.rattle(rattle)
        
        water_box_1.translate([0, 0, 0])
        water_box_2.translate([0, 0, suatoms_max_z + extra_vacuum])
        
        combine_slab = suatoms + water_box_1 + water_box_2
        combine_slab.wrap()
        return combine_slab

    else:
        print("The slab is too thin to add water on top and bottom.")
        return suatoms
slab = read('/work/scratch/as41vomu/PhD_Nov_24/BTO_H2O_Interface/DFT/NEW_CALC_05_03_24/BTO_100_surface_raw_1x1x1_finished/TiO2_relax_3x3x1/CONTCAR')
combine_slab_water = make_sanwich_from_slab(slab, density=0.6)
view(combine_slab_water)
