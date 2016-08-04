import numpy as np, pixie, argparse
from enlib import enmap, powspec, colors, utils
parser = argparse.ArgumentParser()
parser.add_argument("idir", nargs="?", default="inputs")
parser.add_argument("odir", nargs="?", default="imaps")
parser.add_argument("-r", "--res", type=float, default=0.1)
parser.add_argument("-s", "--seed", type=int, default=0)
parser.add_argument("--refsys-res",type=float, default=10)
args = parser.parse_args()

# Generate the inputs to our simulator, which should be enmaps.
# First generate the simulated CMB
res = args.res*utils.degree
shape, wcs = pixie.fullsky_geometry(res, dims=(3,))
np.random.seed(args.seed)

print colors.green + "Simulating reference blackbody" + colors.reset
rshape, rwcs = pixie.fullsky_geometry(args.refsys_res*utils.degree, dims=(3,))
map_ref = pixie.sim_reference_blackbody(rshape, rwcs)
extra = { "NAME": "REFERENCE", "BEAM": "NONE", "SPEC": "BLACK" }
enmap.write_map(args.odir + "/map_ref.fits", map_ref, extra=extra)

# Then project our dust map onto our target coordinates
print colors.green + "Projecting dust model" + colors.reset
heal_dust = pixie.read_healpix(args.idir + "/test_therm_600p0_512_v2.fits")
map_dust  = pixie.project_healpix(shape, wcs, heal_dust, rot="gal,ecl", verbose=True)
extra = {
		"NAME":  "DUST",
		"BEAM":  "GAUSS",
		"FWHM":  0.07,
		"SPEC":  "GRAY",
		"TBODY": 19.6,
		"BETA":  1.59,
		"FREF":  600e9,
		"SUNIT": 1e-20,
	}
enmap.write_map(args.odir + "/map_dust.fits", map_dust, extra=extra)
del heal_dust, map_dust
print

print colors.green + "Simulating CMB" + colors.reset
ps = powspec.read_camb_full_lens(args.idir + "/cl_lensinput.dat")
map_raw, map_lens, map_cmb = pixie.sim_cmb_map(shape, wcs, ps, verbose=True)
extra = { "NAME": "CMB", "BEAM": "NONE", "SPEC": "BLACK" }
enmap.write_map(args.odir + "/map_cmb.fits", map_cmb, extra=extra)
del map_raw, map_lens, map_cmb
print
