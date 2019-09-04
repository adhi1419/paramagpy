import warnings
from random import randint

import numpy as np
from Bio.PDB import PDBParser
from Bio.PDB.Atom import Atom, DisorderedAtom
from Bio.PDB.PDBExceptions import PDBConstructionWarning, PDBConstructionException
from Bio.PDB.Residue import Residue, DisorderedResidue
from Bio.PDB.Structure import Structure
from Bio.PDB.StructureBuilder import StructureBuilder


def rotation_matrix(axis, theta):
    """Return the rotation matrix associated with counterclockwise
    rotation about the given axis by theta radians.

    Parameters
    ----------
    axis : array of floats
        the [x,y,z] axis for rotation.

    Returns
    -------
    matrix : numpy 3x3 matrix object
        the rotation matrix
    """
    axis = np.array(axis)
    axis /= np.linalg.norm(axis)
    a = np.cos(theta / 2.0)
    b, c, d = -axis * np.sin(theta / 2.0)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    return np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                     [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                     [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])


class CustomAtom(Atom):
    MU0 = 4 * np.pi * 1E-7
    HBAR = 1.0546E-34

    gyro_lib = {
        'H': 2 * np.pi * 42.576E6,
        'N': 2 * np.pi * -4.316E6,
        'C': 2 * np.pi * 10.705E6}  # rad/s/T

    csa_lib = {
        'H': (np.array([-5.8, 0.0, 5.8]) * 1E-6, 8. * (np.pi / 180.)),
        'N': (np.array([-62.8, -45.7, 108.5]) * 1E-6, 19. * (np.pi / 180.)),
        'C': (np.array([-86.5, 11.8, 74.7]) * 1E-6, 38. * (np.pi / 180.))}

    valency_lib = {
        'H': 1,
        'N': 3,
        'C': 4,
        'O': 2,
        'S': 2
    }

    """docstring for CustomAtom"""

    def __init__(self, *arg, **kwargs):
        super().__init__(*arg, **kwargs)
        self.coord = np.asarray(self.coord, dtype=np.float64)
        self.gamma = self.gyro_lib.get(self.element, 0.0)
        self._csa = None
        self.valency = self.valency_lib.get(self.element)

    def __repr__(self):
        return "<Atom {0:3d}-{1:}>".format(self.parent.id[1], self.name)

    def top(self):
        return self.parent.parent.parent.parent

    @property
    def position(self):
        return self.coord * 1E-10

    @position.setter
    def position(self, value):
        self.coord = value * 1E10

    @property
    def csa(self):
        """
        Get the CSA tensor at the nuclear position
        This uses the geometry of neighbouring atoms
        and a standard library from Bax J. Am. Chem. Soc. 2000

        Returns
        -------
        matrix : 3x3 array
            the CSA tensor in the PDB frame
            if appropriate nuclear positions are not
            available <None> is returned.
        """

        if self._csa is not None:
            return self._csa

        def norm(x):
            return x / np.linalg.norm(x)

        res = self.parent
        resid = res.id
        respid = resid[0], resid[1] - 1, resid[2]
        resnid = resid[0], resid[1] + 1, resid[2]
        resp = res.parent.child_dict.get(respid)
        resn = res.parent.child_dict.get(resnid)

        pas, beta = self.csa_lib.get(self.name, (None, None))
        if resp:
            Hcond = self.element == 'H', 'N' in res, 'C' in resp, beta
            Ncond = self.element == 'N', 'H' in res, 'C' in resp, beta
        else:
            Hcond = (None,)
            Ncond = (None,)
        if resn:
            Ccond = self.element == 'C', 'H' in resn, 'N' in resn, beta
        else:
            Ccond = (None,)

        if all(Hcond):
            NC_vec = resp['C'].coord - res['N'].coord
            NH_vec = res['H'].coord - res['N'].coord
            z = norm(np.cross(NC_vec, NH_vec))
            R = rotation_matrix(-z, beta)
            x = norm(R.dot(NH_vec))
            y = norm(np.cross(z, x))

        elif all(Ncond):
            NC_vec = resp['C'].coord - res['N'].coord
            NH_vec = res['H'].coord - res['N'].coord
            y = norm(np.cross(NC_vec, NH_vec))
            R = rotation_matrix(-y, beta)
            z = norm(R.dot(NH_vec))
            x = norm(np.cross(y, z))

        elif all(Ccond):
            CN_vec = resn['N'].coord - res['C'].coord
            NH_vec = resn['H'].coord - resn['N'].coord
            x = norm(np.cross(NH_vec, CN_vec))
            R = rotation_matrix(x, beta)
            z = norm(R.dot(CN_vec))
            y = norm(np.cross(z, x))

        else:
            return np.zeros(9).reshape(3, 3)
        transform = np.vstack([x, y, z]).T
        tensor = transform.dot(np.diag(pas)).dot(transform.T)
        return tensor

    @csa.setter
    def csa(self, newTensor):
        if newTensor is None:
            self._csa = None
            return
        try:
            assert newTensor.shape == (3, 3)
        except (AttributeError, AssertionError):
            print("The specified CSA tensor does not have the correct format")
            raise
        self._csa = newTensor

    def dipole_shift_tensor(self, position):
        """
        Calculate the magnetic field shielding tensor at the given postition
        due to the nuclear dipole

        Assumes nuclear spin 1/2

        Parameters
        ----------
        position : array floats
            the position (x, y, z) in meters

        Returns
        -------
        dipole_shielding_tensor : 3x3 array
            the tensor describing magnetic shielding at the given position
        """
        pos = np.array(position, dtype=float) - self.position
        distance = np.linalg.norm(pos)
        preFactor = (self.MU0 * self.gamma * self.HBAR * 0.5) / (4. * np.pi)
        p1 = (1. / distance ** 5) * np.kron(pos, pos).reshape(3, 3)
        p2 = (1. / distance ** 3) * np.identity(3)
        return (preFactor * (3. * p1 - p2))

    def bonded_to(self, valency=None):
        # TO-DO: Documentation
        return self.parent.bonded_to(self, self.valency if valency is None else valency)


class CustomStructure(Structure):
    """This is an overload hack of the BioPython Structure object"""

    def __init__(self, *arg, **kwargs):
        super().__init__(*arg, **kwargs)

    def parse(self, dataValues, models=None):
        used = set([])
        data = []

        if type(models) == int:
            chains = self[models].get_chains()
        elif type(models) in (list, tuple):
            chains = []
            for m in models:
                chains += self[m].get_chains()
        else:
            chains = self.get_chains()

        if dataValues.dtype in ('PCS', 'PRE'):
            for chain in chains:
                for key in dataValues:
                    seq, name = key
                    if seq in chain:
                        resi = chain[seq]
                        if name in resi:
                            atom = resi[name]
                            data.append((atom, *dataValues[key]))
                            used.add(key)

        elif dataValues.dtype in ('RDC', 'CCR'):
            for chain in chains:
                for key in dataValues:
                    (seq1, name1), (seq2, name2) = key
                    if seq1 in chain and seq2 in chain:
                        resi1 = chain[seq1]
                        resi2 = chain[seq2]
                        if name1 in resi1 and name2 in resi2:
                            atom1 = resi1[name1]
                            atom2 = resi2[name2]
                            data.append((atom1, atom2, *dataValues[key]))
                            used.add(key)

        unused = set(dataValues) - used
        if unused:
            message = "WARNING: Some values were not parsed to {}:"
            print(message.format(self.id))
            print(list(unused))
        return data


class CustomResidue(Residue):
    """Paramagpy wrapper for BioPython's Residue entity"""

    def __init__(self, *arg, **kwargs):
        super().__init__(*arg, **kwargs)

    def bonded_to(self, source_atom, valency=None):
        # TO-DO documentation
        def dist(from_atom):
            return np.abs(from_atom - source_atom)

        def quick_select(start, end, val):
            if start >= end:
                return None

            mid = partition(start, end, randint(start, end))
            bucket_a_len = mid - start + 1

            if valency < bucket_a_len:
                quick_select(start, mid - 1, val)
            elif valency > bucket_a_len:
                quick_select(mid + 1, end, val - bucket_a_len)

        def partition(start, end, rnd):
            atoms[start], atoms[rnd] = atoms[rnd], atoms[start]
            _start = start
            pivot = dist(atoms[start])

            start += 1
            while True:
                while start < end and dist(atoms[start]) < pivot:
                    start += 1
                while start <= end and dist(atoms[end]) >= pivot:
                    end -= 1
                if start >= end:
                    break
                atoms[start], atoms[end] = atoms[end], atoms[start]

            atoms[_start], atoms[end] = atoms[end], atoms[_start]
            return end

        # If id is supplied instead of CustomAtom, convert accordingly
        if type(source_atom) is str:
            source_atom = self[source_atom] if self.has_id(source_atom) else None

        # Error handling

        # If source_atom is not part of the residue, raise an exception
        if source_atom not in self.child_list:
            raise ValueError("source_atom is not part of this residue, call this function with a valid source")

        # If valency isn't provided, use the CustomAtom instance's valency
        # If still not available, throw an exception (or possibly return atoms at distance < 1.8E-10)
        valency = source_atom.valency if valency is None else valency
        if valency is None:
            raise ValueError(
                "Couldn't obtain valency for this source atom, call this function with the valency parameter")

        # Only look for the nearest atoms at distance < 1.8E-10
        # dist > 0 ensures the source_atom itself isn't considered
        atoms = [atom for atom in self.get_atoms() if 0 < dist(atom) < 1.8]
        quick_select(0, len(atoms) - 1, valency)
        return atoms[:valency]


class CustomStructureBuilder(StructureBuilder):
    """This is an overload hack of BioPython's CustomStructureBuilder"""

    def __init__(self, *arg, **kwargs):
        super().__init__(*arg, **kwargs)

    def init_structure(self, structure_id):
        self.structure = CustomStructure(structure_id)

    def init_residue(self, resname, field, resseq, icode):
        """Create a new Residue object.

        Arguments:
         - resname - string, e.g. "ASN"
         - field - hetero flag, "W" for waters, "H" for
           hetero residues, otherwise blank.
         - resseq - int, sequence identifier
         - icode - string, insertion code

        """
        if field != " ":
            if field == "H":
                # The hetero field consists of H_ + the residue name (e.g. H_FUC)
                field = "H_" + resname
        res_id = (field, resseq, icode)
        if field == " ":
            if self.chain.has_id(res_id):
                # There already is a residue with the id (field, resseq, icode).
                # This only makes sense in the case of a point mutation.
                warnings.warn("WARNING: Residue ('%s', %i, '%s') "
                              "redefined at line %i."
                              % (field, resseq, icode, self.line_counter),
                              PDBConstructionWarning)
                duplicate_residue = self.chain[res_id]
                if duplicate_residue.is_disordered() == 2:
                    # The residue in the chain is a DisorderedResidue object.
                    # So just add the last Residue object.
                    if duplicate_residue.disordered_has_id(resname):
                        # The residue was already made
                        self.residue = duplicate_residue
                        duplicate_residue.disordered_select(resname)
                    else:
                        # Make a new residue and add it to the already
                        # present DisorderedResidue
                        new_residue = CustomResidue(res_id, resname, self.segid)
                        duplicate_residue.disordered_add(new_residue)
                        self.residue = duplicate_residue
                        return
                else:
                    if resname == duplicate_residue.resname:
                        warnings.warn("WARNING: Residue ('%s', %i, '%s','%s')"
                                      " already defined with the same name "
                                      "at line  %i."
                                      % (field, resseq, icode, resname,
                                         self.line_counter),
                                      PDBConstructionWarning)
                        self.residue = duplicate_residue
                        return
                    # Make a new DisorderedResidue object and put all
                    # the Residue objects with the id (field, resseq, icode) in it.
                    # These residues each should have non-blank altlocs for all their atoms.
                    # If not, the PDB file probably contains an error.
                    if not self._is_completely_disordered(duplicate_residue):
                        # if this exception is ignored, a residue will be missing
                        self.residue = None
                        raise PDBConstructionException(
                            "Blank altlocs in duplicate residue %s ('%s', %i, '%s')"
                            % (resname, field, resseq, icode))
                    self.chain.detach_child(res_id)
                    new_residue = CustomResidue(res_id, resname, self.segid)
                    disordered_residue = DisorderedResidue(res_id)
                    self.chain.add(disordered_residue)
                    disordered_residue.disordered_add(duplicate_residue)
                    disordered_residue.disordered_add(new_residue)
                    self.residue = disordered_residue
                    return
        self.residue = CustomResidue(res_id, resname, self.segid)
        self.chain.add(self.residue)

    def init_atom(self, name, coord, b_factor, occupancy, altloc, fullname,
                  serial_number=None, element=None):
        """Create a new Atom object.
        Arguments:
         - name - string, atom name, e.g. CA, spaces should be stripped
         - coord - Numeric array (Float0, size 3), atomic coordinates
         - b_factor - float, B factor
         - occupancy - float
         - altloc - string, alternative location specifier
         - fullname - string, atom name including spaces, e.g. " CA "
         - element - string, upper case, e.g. "HG" for mercury
        """
        residue = self.residue
        # if residue is None, an exception was generated during
        # the construction of the residue
        if residue is None:
            return
        # First check if this atom is already present in the residue.
        # If it is, it might be due to the fact that the two atoms have atom
        # names that differ only in spaces (e.g. "CA.." and ".CA.",
        # where the dots are spaces). If that is so, use all spaces
        # in the atom name of the current atom.
        if residue.has_id(name):
            duplicate_atom = residue[name]
            # atom name with spaces of duplicate atom
            duplicate_fullname = duplicate_atom.get_fullname()
            if duplicate_fullname != fullname:
                # name of current atom now includes spaces
                name = fullname
                warnings.warn("Atom names %r and %r differ "
                              "only in spaces at line %i."
                              % (duplicate_fullname, fullname,
                                 self.line_counter),
                              PDBConstructionWarning)
        self.atom = CustomAtom(name, coord, b_factor, occupancy, altloc,
                               fullname, serial_number, element)
        if altloc != " ":
            # The atom is disordered
            if residue.has_id(name):
                # Residue already contains this atom
                duplicate_atom = residue[name]
                if duplicate_atom.is_disordered() == 2:
                    duplicate_atom.disordered_add(self.atom)
                else:
                    # This is an error in the PDB file:
                    # a disordered atom is found with a blank altloc
                    # Detach the duplicate atom, and put it in a
                    # DisorderedAtom object together with the current
                    # atom.
                    residue.detach_child(name)
                    disordered_atom = DisorderedAtom(name)
                    residue.add(disordered_atom)
                    disordered_atom.disordered_add(self.atom)
                    disordered_atom.disordered_add(duplicate_atom)
                    residue.flag_disordered()
                    warnings.warn("WARNING: disordered atom found "
                                  "with blank altloc before line %i.\n"
                                  % self.line_counter,
                                  PDBConstructionWarning)
            else:
                # The residue does not contain this disordered atom
                # so we create a new one.
                disordered_atom = DisorderedAtom(name)
                residue.add(disordered_atom)
                # Add the real atom to the disordered atom, and the
                # disordered atom to the residue
                disordered_atom.disordered_add(self.atom)
                residue.flag_disordered()
        else:
            # The atom is not disordered
            residue.add(self.atom)


def load_pdb(fileName, ident=None):
    """
    Read PDB from file into biopython structure object

    Parameters
    ----------
    fileName : str
        the path to the file
    ident : str (optional)
        the desired identity of the structure object

    Returns
    -------
    values : :class:`paramagpy.protein.CustomStructure`
        a structure object containing the atomic coordinates
    """
    if not ident:
        ident = fileName
    parser = PDBParser(structure_builder=CustomStructureBuilder())
    return parser.get_structure(ident, fileName)
