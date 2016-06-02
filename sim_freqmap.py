import numpy as np, argparse, pixie
from enlib import enmap, utils
parser = argparse.ArgumentParser()
parser.add_argument("freqs")
parser.add_argument("omap")
parser.add_argument("-C", "--config",   type=str, default=None)
parser.add_argument("-S", "--sky",      type=int, default=0)
parser.add_argument("-T", "--template", type=str, default=None)
parser.add_argument("-u", "--unit",     type=str, default=None)
args = parser.parse_args()

config = pixie.load_config(args.config)
freqs  = pixie.parse_floats(args.freqs)

# Load the fields of our chosen sky
sky    = config.skies[args.sky]
fields = [pixie.read_field(field) for field in sky]

if args.template:
	templ = enmap.read_map(args.template)
	shape, wcs = templ.shape, templ.wcs
else:
	shape, wcs = pixie.fullsky_geometry(res=config.patch_res*utils.degree)

maps = [field.project(shape, wcs)(freqs) for field in fields]
maps.insert(0, np.sum(maps,0))
maps = enmap.samewcs(np.array(maps), maps[1])

if args.unit:
	if args.unit == "cmb":
		# CMB uK equivalent
		scale = pixie.blackbody(freqs, pixie.Tcmb)
		maps *= pixie.Tcmb * 1e6 / scale[:,None,None,None]
	else:
		raise ValueError("Unknown unit '%s'" % args.unit)

# And write our results
enmap.write_map(args.omap, maps)
