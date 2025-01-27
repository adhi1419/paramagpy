# PyMOL macro for loading tensor isosurface from paramagpy
# ax    | 1E-32 m^3 :    -8.688
# rh    | 1E-32 m^3 :    -4.192
# x     |   1E-10 m :    25.517
# y     |   1E-10 m :     8.652
# z     |   1E-10 m :     6.358
# a     |       deg :   116.011
# b     |       deg :   138.058
# g     |       deg :    43.493
# mueff |        Bm :     9.581
# shift |       ppm :     0.000
# B0    |         T :    18.790
# temp  |         K :   298.150
# t1e   |        ps :     0.189
# taur  |        ns :     4.100

import os, pymol
curdir = os.path.dirname(pymol.__script__)
set normalize_ccp4_maps, off
meshfile = os.path.join(curdir, 'isosurface.pml.ccp4')
cmd.load(meshfile, 'isomap', 1, 'ccp4')
isosurface pos_isosurface, isomap, 50.0
isosurface neg_isosurface, isomap, -50.0
set transparency, 0.5, pos_isosurface
set transparency, 0.5, neg_isosurface
set surface_color, blue, pos_isosurface
set surface_color, red, neg_isosurface
pseudoatom ori_isosurface, pos=[25.517000000000003, 8.652, 6.3580000000000005]
show spheres, ori_isosurface
color pink, ori_isosurface
cmd.load(os.path.join(curdir, '4icbH_mut.pdb'),'4icbH_mut')
show_as cartoon, 4icbH_mut
