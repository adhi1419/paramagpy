import numpy as np

from paramagpy import protein


class Tracker:

    def __init__(self, rs):
        self.rs = rs
        self.res_atoms = {}

    def save_atom_coords(self):
        for x in self.rs.get_atoms():
            if self.res_atoms.get(x) is None:
                self.res_atoms[x] = []
            self.res_atoms[x] += [x.get_coord()]

    def print_atom_coords(self):
        for atom in self.res_atoms:
            print(f"{atom} : {self.res_atoms[atom]}")

    def print_atom_linkage(self):
        atoms = list(self.rs.get_atoms())
        for atom in atoms:
            print(f"{atom} : {[x for x in atom.bonded_to()]}")


def rad(deg): return (deg / 180) * np.pi


prot = protein.load_pdb('../data_files/lys.pdb')

res = prot[0]['A'][55]

trk = Tracker(res)

trk.print_atom_linkage()
trk.save_atom_coords()

# Default: [rad(47.63), rad(162.49), rad(-176.18), rad(141.70)
res.set_dihedral(np.array([rad(37.63), rad(162.49), rad(-176.18), rad(146.70)]))
# Will give back original if the below line is executed
# res.set_delta_dihedral(np.array([rad(10), rad(0), rad(0), rad(-5)]))

trk.save_atom_coords()
trk.print_atom_coords()

from Bio.PDB.mmcifio import MMCIFIO

io = MMCIFIO()
io.set_structure(prot)
io.save("bio-pdb-mmcifio-out.cif")
