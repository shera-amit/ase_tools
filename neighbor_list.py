import torch
import numpy as np
from ase import Atoms
from ase.io import read
import time

class OptimizedPeriodicBoundaryConditions:
    def __init__(self, atoms, cutoff):
        # cpu device
        # device = torch.device('cpu')
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.positions = torch.tensor(atoms.get_positions(), dtype=torch.float32).to(device)
        self.box_size = torch.tensor(atoms.get_cell().lengths(), dtype=torch.float32).to(device)
        self.symbols = atoms.get_chemical_symbols()
        self.cutoff = cutoff
        self.num_atoms = len(atoms)
        self.device = device

    def apply_pbc(self, distance):
        """Apply periodic boundary conditions to distance."""
        return distance - self.box_size * torch.round(distance / self.box_size)

    def compute_distances(self):
        """Compute all pairwise distances with PBC."""
        diff = self.positions.unsqueeze(0) - self.positions.unsqueeze(1)
        diff_pbc = self.apply_pbc(diff)
        distances = torch.norm(diff_pbc, dim=-1)
        return distances, diff_pbc

    def build_neighbor_list(self):
        """Build neighbor list considering periodic boundary conditions."""
        distances, diff_pbc = self.compute_distances()
        mask = (distances < self.cutoff) & (distances > 0)
        
        neighbor_indices = mask.nonzero(as_tuple=False)
        neighbor_distances = distances[mask]
        neighbor_vectors = diff_pbc[mask]

        # Sort neighbors by distance for each atom
        sorted_indices = torch.argsort(neighbor_distances)
        neighbor_indices = neighbor_indices[sorted_indices]
        neighbor_distances = neighbor_distances[sorted_indices]
        neighbor_vectors = neighbor_vectors[sorted_indices]

        # Create output arrays
        max_neighbors = torch.max(torch.sum(mask, dim=1)).item()
        neighbor_list = np.full((self.num_atoms, max_neighbors), -1, dtype=np.int64)
        distance_list = np.zeros((self.num_atoms, max_neighbors), dtype=np.float32)
        vector_list = np.zeros((self.num_atoms, max_neighbors, 3), dtype=np.float32)

        # Fill the output arrays
        for i in range(self.num_atoms):
            neighbors = neighbor_indices[neighbor_indices[:, 0] == i, 1]
            distances = neighbor_distances[neighbor_indices[:, 0] == i]
            vectors = neighbor_vectors[neighbor_indices[:, 0] == i]
            
            num_neighbors = len(neighbors)
            neighbor_list[i, :num_neighbors] = neighbors.cpu().numpy()
            distance_list[i, :num_neighbors] = distances.cpu().numpy()
            vector_list[i, :num_neighbors] = vectors.cpu().numpy()

        return {
            'neighbor_list': neighbor_list,
            'distance_list': distance_list,
            'vector_list': vector_list,
            'positions': self.positions.cpu().numpy(),
            'symbols': np.array(self.symbols)
        }
