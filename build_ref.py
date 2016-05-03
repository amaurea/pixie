import numpy as np, argparse, pixutils, enlib.wcs
from enlib import enmap, utils
parser = argparse.ArgumentParser()
parser.add_argument("ofile")
parser.add_argument("-T", "--temperature", type=float, default=2.725)
parser.add_argument("-n", "--nfreq",  type=int,   default=512)
parser.add_argument("-F", "--fmax",   type=float, default=6e12)
parser.add_argument("-u", "--ounit",  type=float, default=1e-20)
parser.add_argument("-D", "--delaymap", action="store_true")
args = parser.parse_args()

nfreq = args.nfreq
freqs = np.arange(args.nfreq)*args.fmax/args.nfreq
ounit = args.ounit
spec_wcs = enlib.wcs.WCS(naxis=1)
spec_wcs.wcs.cdelt[0] = freqs[1]
spec_wcs.wcs.ctype[0] = 'FREQ'

ospec  = pixutils.blackbody(freqs, [args.temperature,0,0])
ospec /= ounit

if args.delaymap:
	# Go from spectrum to delay (spectrogram)
	ospec, spec_wcs = pixutils.spec2delay(ospec, spec_wcs, axis=0)
# Reorder to TQU,freq
ospec = np.rollaxis(ospec,1)
# And add dummy pixel axes
ospec = pixutils.arr2fullsky(ospec)

pixutils.write_map(args.ofile, ospec, spec_wcs)
