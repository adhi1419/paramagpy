from numpy import pi

from paramagpy import protein


class Tracker:
    res_atoms = {}

    def __init__(self, rs):
        self.rs = rs

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


def rad(deg): return (deg / 180) * pi


prot = protein.load_pdb('../data_files/4icbH_mut.pdb')
res = prot[0]['A'][55]

trk = Tracker(res)

trk.print_atom_linkage()
trk.save_atom_coords()
res.set_rotamer([rad(18), rad(30), rad(0), rad(65)])
trk.save_atom_coords()
trk.print_atom_coords()
trk.print_atom_linkage()
