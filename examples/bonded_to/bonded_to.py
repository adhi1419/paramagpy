import time

import numpy as np
from tabulate import tabulate

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


rot = rad(10)

timer = {'start_program': time.perf_counter() * 1E3, 'start_pdb_parser': time.perf_counter() * 1E3}
prot = protein.load_pdb('../data_files/lys.pdb')
timer['end_pdb_parser'] = time.perf_counter() * 1E3

res = prot[0]['A'][55]

trk = Tracker(res)

timer['start_atom_linkage'] = time.perf_counter() * 1E3
trk.print_atom_linkage()
timer['end_atom_linkage'] = time.perf_counter() * 1E3
trk.save_atom_coords()

# Default: [rad(47.63), rad(162.49), rad(-176.18), rad(141.70)
timer['start_set_dihedral'] = time.perf_counter() * 1E3
res.set_dihedral(np.array([rad(37.63), rad(162.49), rad(-176.18), rad(146.70)]))
timer['end_set_dihedral'] = time.perf_counter() * 1E3
# Will give back original if the below line is executed
timer['start_set_delta_dihedral'] = time.perf_counter() * 1E3
res.set_delta_dihedral(np.array([rad(10), 0, 0, rad(-5)]))
timer['end_set_delta_dihedral'] = time.perf_counter() * 1E3
res.set_delta_dihedral(np.array([rot, rot, rot, rot]))

# Full sweep

import cProfile

pr = cProfile.Profile()
pr.enable()
timer['start_set_delta_dihedral_full'] = time.perf_counter() * 1E3
res.grid_search_rotamer(rot)
timer['end_set_delta_dihedral_full'] = time.perf_counter() * 1E3
pr.disable()
# after your program ends
pr.print_stats(sort="tottime")

trk.save_atom_coords()
trk.print_atom_coords()
timer['end_program'] = time.perf_counter() * 1E3

print()
print()
print('='*50)
print('Benchmark Results')
print('='*50)
print(tabulate([['Program', timer['end_program'] - timer['start_program']],
                ['PDB Parsing', timer['end_pdb_parser'] - timer['start_pdb_parser']],
                ['Atom Linkages', timer['end_atom_linkage'] - timer['start_atom_linkage']],
                ['Set Dihedral', timer['end_set_dihedral'] - timer['start_set_dihedral']],
                ['Set delta-dihedral', timer['end_set_delta_dihedral'] - timer['start_set_delta_dihedral']],
                ['Set delta-dihedral Full', timer['end_set_delta_dihedral_full'] - timer['start_set_delta_dihedral_full']]],
               headers=['Name', 'Duration (ms)'], tablefmt="fancy_grid"))

from Bio.PDB.mmcifio import MMCIFIO

io = MMCIFIO()
io.set_structure(prot)
io.save("bio-pdb-mmcifio-out.cif")
