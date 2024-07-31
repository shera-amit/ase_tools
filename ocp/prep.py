from fairchem.data.oc.core import (
    Adsorbate,
    Bulk,
    Slab,
    MultipleAdsorbateSlabConfig,
    AdsorbateSlabConfig,
)
from ase.io import read
from ase import Atoms
from ase.db import connect

db = connect("test.db")
atoms = read("./bto_cubic.cif")

index = [(1, 0, 0), (1, 1, 1), (1, 1, 0)]
# create a water molecule using ase build
water = Atoms("H2O", positions=[[0, 0, 0], [0.95, 0, 0], [0.1587, 0.6, 0]])
for indx in index:
    bulk = Bulk(atoms)
    adsorbate = Adsorbate(water)
    slabs = Slab.from_bulk_get_specific_millers(bulk=bulk, specific_millers=indx)

    slab = slabs[0]

    adsorbate_slab_config = AdsorbateSlabConfig(
        slab, adsorbate, mode="random", num_sites=100, num_augmentations_per_site=2
    )
    atoms_list = adsorbate_slab_config.atoms_list
    for at in atoms_list:
        db.write(at, data={"index": indx})

    atoms_list = []
    for i in range(100):
        multi_ads = MultipleAdsorbateSlabConfig(
            slab, adsorbates=[adsorbate, adsorbate], mode="random"
        )
        atoms_list.append(multi_ads.atoms_list)

    for at in atoms_list:
        db.write(at[0], data={"index": indx})

for row in db.select():
    atoms = row.toatoms()
    len_atoms = len(atoms)
    # remove constraints
    atoms.set_constraint()

    db.write(atoms, id=row.id)
