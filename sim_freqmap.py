import numpy as np, argparse, pixie
from enlib import enmap, utils, sharp
parser = argparse.ArgumentParser()
parser.add_argument("freqs")
parser.add_argument("omap")
parser.add_argument("-C", "--config",   type=str, default=None)
parser.add_argument("-S", "--sky",      type=int, default=0)
parser.add_argument("-T", "--template", type=str, default=None)
parser.add_argument("-u", "--unit",     type=str, default=None)
parser.add_argument("-B", "--apply-beam", type=int, default=0)
args = parser.parse_args()

config = pixie.load_config(args.config)
freqs  = pixie.parse_floats(args.freqs)
beam_lmax = 1000
bsigma  = 1.9*utils.degree*utils.fwhm
fscatter= 1.5e12
scatter = np.exp(-(freqs/fscatter)**2)

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
	# We will apply a hardcoded gaussian for now
	l       = np.arange(beam_lmax+1)
	beam    = np.exp(-0.5*l*(l+1)*bsigma**2)
	bmat    = np.zeros((3,3,len(beam)))
	for i in range(3): bmat[i,i] = beam
	# Smooth manually using full-sky geometry
	for field in fields:
		print field.name
		# Make sure we have the standard pixel ordering before transforming
		if field.map.wcs.wcs.cdelt[0] > 0: field.map = field.map[...,:,::-1]
		if field.map.wcs.wcs.cdelt[1] < 0: field.map = field.map[...,::-1,:]
		field.map = enmap.samewcs(np.ascontiguousarray(field.map), field.map)
		minfo = sharp.map_info_clenshaw_curtis(field.map.shape[-2], field.map.shape[-1])
		ainfo = sharp.alm_info(lmax=beam_lmax)
		sht   = sharp.sht(minfo, ainfo)
		alm   = np.zeros((3,ainfo.nelem),dtype=complex)
		print "T -> alm"
		sht.map2alm(field.map[:1].reshape(1,-1), alm[:1])
		print "P -> alm"
		sht.map2alm(field.map[1:].reshape(2,-1), alm[1:], spin=2)
		print "lmul"
		alm   = ainfo.lmul(alm, bmat)
		# And transform back again
		print "alm -> T"
		sht.alm2map(alm[:1], field.map[:1].reshape(1,-1))
		print "alm -> P"
		sht.alm2map(alm[1:], field.map[1:].reshape(2,-1), spin=2)
		# Reapply spline filter
		print "Prefilter"
		field.pmap = utils.interpol_prefilter(field.map, order=field.order)
	print "Beam done"

maps = [field.project(shape, wcs)(freqs) for field in fields]
if args.apply_beam:
	maps = [map*scatter[:,None,None,None] for map in maps]
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
