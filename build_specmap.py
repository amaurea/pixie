import numpy as np, argparse, pixutils, enlib.wcs
from enlib import enmap, utils
parser = argparse.ArgumentParser()
parser.add_argument("idescs", nargs="+")
parser.add_argument("omap")
parser.add_argument("-n", "--nfreq",  type=int,   default=512)
parser.add_argument("-F", "--fmax",   type=float, default=6e12)
parser.add_argument("-b", "--blocksize", type=int, default=16)
parser.add_argument("-u", "--ounit",  type=float, default=1e-20)
parser.add_argument("-D", "--delaymap", action="store_true")
args = parser.parse_args()

nfreq = args.nfreq
bsize = args.blocksize
freqs = np.arange(args.nfreq)*args.fmax/args.nfreq
ounit = args.ounit
spec_wcs = enlib.wcs.WCS(naxis=1)
spec_wcs.wcs.cdelt[0] = freqs[1]
spec_wcs.wcs.ctype[0] = 'FREQ'

# Build spectrum blocks
nblock  = (nfreq+bsize-1)/bsize
fblocks = np.minimum(np.array([
	np.arange(0, nblock)*bsize,
	np.arange(1,nblock+1)*bsize]).T, nfreq)

def ftok(toks, i, default):
	return float(default) if len(toks) <= i else float(toks[i])

ospec = None
for idesc in args.idescs:
	print idesc
	toks  = idesc.split(":")
	spectype, ifile = toks[:2]
	m = enmap.read_map(ifile)
	if ospec is None:
		ospec = enmap.zeros((nfreq,)+m.shape, m.wcs)
	if spectype in ["cmb"]:
		Tbody = ftok(toks, 2, 2.725)
		unit  = ftok(toks, 3, 1e-6)
		for f1,f2 in fblocks:
			mean  = pixutils.blackbody(freqs[f1:f2], [Tbody,0,0])[:,:,None,None]
			slope = pixutils.blackbody(freqs[f1:f2], Tbody, deriv=True)[:,None,None,None] * unit
			ospec[f1:f2] += mean + slope * m
			del mean, slope
	elif spectype in ["dust"]:
		Tbody = ftok(toks, 2, 19.6)
		beta  = ftok(toks, 3, 1.59)
		fref  = ftok(toks, 4, 600e9)
		unit  = ftok(toks, 5, 1e-20)
		scale = pixutils.graybody(freqs, Tbody, beta) / pixutils.graybody(fref, Tbody, beta) * unit
		for f1,f2 in fblocks:
			ospec[f1:f2] += scale[f1:f2,None,None,None] * m[None,:,:,:]
ospec /= ounit

if args.delaymap:
	# Go from spectrum to delay (spectrogram)
	ospec, spec_wcs = pixutils.spec2delay(ospec, spec_wcs, axis=0)

# Reorder to TQU,freq,y,x
ospec = np.rollaxis(ospec,1)

pixutils.write_map(args.omap, ospec, spec_wcs)
