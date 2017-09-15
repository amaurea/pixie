import numpy as np, argparse, pixie
from enlib import enmap, utils, sharp
parser = argparse.ArgumentParser()
parser.add_argument("freqs")
parser.add_argument("omap")
parser.add_argument("-C", "--config",   type=str, default=None)
parser.add_argument("-S", "--sky",      type=int, default=0)
parser.add_argument("-T", "--template", type=str, default=None)
parser.add_argument("-u", "--unit",     type=str, default=None)
parser.add_argument("-B", "--apply-beam", type=int, default=1)
args = parser.parse_args()

config = pixie.load_config(args.config)
freqs  = pixie.parse_floats(args.freqs)
print freqs
beam_lmax = 1000
#bsigma  = 1.9*utils.degree*utils.fwhm
beam    = pixie.BeamGauss(1.9*utils.degree*utils.fwhm)
#fscatter= 1.5e12
#scatter = np.exp(-(freqs/fscatter)**2)

# Load the fields of our chosen sky
sky    = config.skies[args.sky]
fields = [pixie.read_field(field) for field in sky]

if args.template:
	templ = enmap.read_map(args.template)
	shape, wcs = templ.shape, templ.wcs
else:
	shape, wcs = pixie.fullsky_geometry(res=config.patch_res*utils.degree)

if args.apply_beam:
	print "Applying beam"
	for i, field in enumerate(fields):
		print field.name
		fields[i] = field.to_beam(beam)
	print "Beam done"

posmap = enmap.posmap(shape, wcs)
maps = [[field.at(freq, posmap) for freq in freqs] for field in fields]
maps.insert(0, np.sum(maps,0))
maps = enmap.ndmap(np.array(maps), wcs)

# We now have [comp,freq,stokes,y,x]

if args.unit:
	if args.unit == "cmb":
		# CMB uK equivalent
		scale = pixie.blackbody(freqs, pixie.Tcmb)
		maps *= pixie.Tcmb * 1e6 / scale[:,None,None,None]
	else:
		raise ValueError("Unknown unit '%s'" % args.unit)

# And write our results
enmap.write_map(args.omap, maps)
