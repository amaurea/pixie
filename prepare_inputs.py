import numpy as np, pixutils, argparse
from enlib import enmap, powspec, colors, utils
parser = argparse.ArgumentParser()
parser.add_argument("idir", nargs="?", default="inputs")
parser.add_argument("odir", nargs="?", default="imaps")
args = parser.parse_args()

# Generate the inputs to our simulator, which should be enmaps.
# First generate the simulated CMB
shape, wcs = pixutils.fullsky_geometry(0.1*utils.degree, dims=(3,))

print colors.green + "Simulating reference blackbody" + colors.reset
rshape, rwcs = pixutils.fullsky_geometry(np.pi/2, dims=(3,))
map_ref = pixutils.sim_reference_blackbody(rshape, rwcs)
enmap.write_map(args.odir + "/map_ref.fits", map_ref)

# Then project our dust map onto our target coordinates
print colors.green + "Projecting dust model" + colors.reset
heal_dust = pixutils.read_healpix(args.idir + "/test_therm_600p0_512_v2.fits")
map_dust  = pixutils.project_healpix(shape, wcs, heal_dust, rot="gal,ecl", verbose=True)
enmap.write_map(args.odir + "/map_dust.fits", map_dust)
del heal_dust, map_dust
print

print colors.green + "Simulating CMB" + colors.reset
ps = powspec.read_camb_full_lens(args.idir + "/cl_lensinput.dat")
map_raw, map_lens, map_cmb = pixutils.sim_cmb_map(shape, wcs, ps, verbose=True)
enmap.write_map(args.odir + "/map_cmb.fits", map_cmb)
del map_raw, map_lens, map_cmb
print
