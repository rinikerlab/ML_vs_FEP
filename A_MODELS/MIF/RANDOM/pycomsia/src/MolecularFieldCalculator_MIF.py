""""
This code has been adapted from Pycomsia (See PYCOMSIA_LICENSE.txt).
"""

""""
Copyright (C) 2025 ETH Zurich, Stefan J.P. Chrin, Gregory Landrum, Sereina Riniker
"""


from rdkit import Chem
from rdkit.Chem import rdMIF
from rdkit.Geometry import rdGeometry as geom
import numpy as np

class MolecularFieldCalculator_MIF:
    """
    This version replaces the original manual field calculations
    with rdMIF classes:
        - UFFVdWaals    -> steric_field
        - CoulombDielectric (or Coulomb) -> electrostatic_field
        - Hydrophilic   -> hydrophobic_field (inverted usage, see note)
        - HBond         -> hbond_donor_field / hbond_acceptor_field
          (we use different probeAtomTypes for each, for illustration)
    
    The rest of the script (main.py) remains unchanged since
    we still return the same dictionary structure containing
    'train_fields' and 'pred_fields' with keys:
        steric_field, electrostatic_field, hydrophobic_field,
        hbond_donor_field, hbond_acceptor_field
    Each entry is a list of flattened numpy arrays, one per molecule.
    """

    def __init__(self):
        # No manual alpha/tolerance needed; rdMIF descriptors handle it
        pass

    def calc_field(self, aligned_results, grid_spacing, grid_dimensions, grid_origin):
        """
        Calculate molecular fields for both training and prediction sets,
        but now using rdMIF descriptors.

        Returns a dict:
        {
          'train_fields': {
              'steric_field': [...],
              'electrostatic_field': [...],
              'hydrophobic_field': [...],
              'hbond_donor_field': [...],
              'hbond_acceptor_field': [...]
          },
          'pred_fields': {
              'steric_field': [...],
              'electrostatic_field': [...],
              'hydrophobic_field': [...],
              'hbond_donor_field': [...],
              'hbond_acceptor_field': [...]
          }
        }
        """
        # Prepare output dictionaries
        train_fields = {
            'steric_field': [],
            'electrostatic_field': [],
            'hydrophobic_field': [],
            'hbond_donor_field': [],
            'hbond_acceptor_field': []
        }
        pred_fields = {
            'steric_field': [],
            'electrostatic_field': [],
            'hydrophobic_field': [],
            'hbond_donor_field': [],
            'hbond_acceptor_field': []
        }

        # Process each molecule
        for mol, is_training in aligned_results:
            if mol is not None:
                # Generate all 5 fields for a single molecule
                fields = self._calc_single_molecule_fields(
                    mol, grid_spacing, grid_dimensions, grid_origin
                )
                # Append each field's flattened data
                target_fields = train_fields if is_training else pred_fields
                for field_name in target_fields:
                    target_fields[field_name].append(fields[field_name].flatten())

        return {'train_fields': train_fields, 'pred_fields': pred_fields}

    def _calc_single_molecule_fields(self, mol, grid_spacing, grid_dimensions, grid_origin):
        """
        Use rdMIF descriptors to compute and return a dictionary with
        'steric_field', 'electrostatic_field', 'hydrophobic_field',
        'hbond_donor_field', 'hbond_acceptor_field'.
        """
        
        from rdkit.Chem import AllChem
        AllChem.ComputeGasteigerCharges(mol)
        
        dx, dy, dz = grid_spacing
        nx, ny, nz = grid_dimensions
        x0, y0, z0 = grid_origin


        # Create placeholders
        steric = np.zeros((nx, ny, nz), dtype=float)
        electrostatic = np.zeros((nx, ny, nz), dtype=float)
        hydrophobic = np.zeros((nx, ny, nz), dtype=float)
        hbond_donor = np.zeros((nx, ny, nz), dtype=float)
        hbond_acceptor = np.zeros((nx, ny, nz), dtype=float)

        # Create a UniformRealValueGrid3D for each field, so we can
        # fill it with the corresponding rdMIF descriptor values.
        # Then we extract them into a NumPy array.

        # Helper to build an empty grid matching the bounding box
        def build_grid():
            return geom.UniformRealValueGrid3D(
                int(nx), int(ny), int(nz),
                float(dx),
                geom.Point3D(x0, y0, z0)
            )

        # 1) steric_field -> UFFVdWaals (e.g. "probeAtomType='C_3'")
        grid_steric = build_grid()
        steric_descriptor = rdMIF.UFFVdWaals(mol)
        rdMIF.CalculateDescriptors(grid_steric, steric_descriptor)
        steric = self._grid_to_numpy(grid_steric, (nx, ny, nz))

        # 2) electrostatic_field -> CoulombDielectric
        grid_electro = build_grid()
        electro_descriptor = rdMIF.CoulombDielectric(mol)
        rdMIF.CalculateDescriptors(grid_electro, electro_descriptor)
        electrostatic = self._grid_to_numpy(grid_electro, (nx, ny, nz))

        # 3) hydrophobic_field -> use "Hydrophilic" MIF
        #    but we invert it to approximate hydrophobic. (Optional or skip the inversion)
        grid_hydro = build_grid()
        hydro_descriptor = rdMIF.Hydrophilic(mol)
        rdMIF.CalculateDescriptors(grid_hydro, hydro_descriptor)
        # For purely "hydrophobic," you might invert or negative it. We'll keep it as-is.
        hydrophobic_vals = self._grid_to_numpy(grid_hydro, (nx, ny, nz))
        hydrophobic = hydrophobic_vals  # or e.g. -1.0 * hydrophobic_vals

        # 4) hbond_donor_field -> "HBond" with a donor-like probe
        grid_donor = build_grid()
        # Example: "O" or "OH" or "NH" for probeAtomType to simulate donor or acceptor
        hbond_donor_descriptor = rdMIF.HBond(mol, probeAtomType="NH", fixed=False)
        rdMIF.CalculateDescriptors(grid_donor, hbond_donor_descriptor)
        hbond_donor = self._grid_to_numpy(grid_donor, (nx, ny, nz))

        # 5) hbond_acceptor_field -> "HBond" with an acceptor-like probe
        grid_acceptor = build_grid()
        hbond_acceptor_descriptor = rdMIF.HBond(mol, probeAtomType="O", fixed=False)
        rdMIF.CalculateDescriptors(grid_acceptor, hbond_acceptor_descriptor)
        hbond_acceptor = self._grid_to_numpy(grid_acceptor, (nx, ny, nz))

        # Return in the same structure used by the main script
        return {
            'steric_field': steric,
            'electrostatic_field': electrostatic,
            'hydrophobic_field': hydrophobic,
            'hbond_donor_field': hbond_donor,
            'hbond_acceptor_field': hbond_acceptor
        }

    def _grid_to_numpy(self, rdkit_grid, shape):
        import numpy as np
        # Calculate total number of grid points from shape (nx, ny, nz)
        n_points = np.prod(shape)
        values = np.zeros(n_points, dtype=float)
        # Fix: iterate from 0 to n_points-1 (instead of an off-by-one range)
        for i in range(n_points):
            values[i] = rdkit_grid.GetVal(i)
        return values.reshape(shape)
