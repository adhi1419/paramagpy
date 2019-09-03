from paramagpy import protein

prot = protein.load_pdb('/home/go/Workspace/paramagpy_extra/4icbH_mut_1.pdb')
res = prot[0]['A'][10]
atoms = list(res.get_atoms())

print(res)
print(atoms)

print_str = "{} : {}"
for atom in atoms:
    print(print_str.format(atom, {x: x - atom for x in atom.bonded_to()}))