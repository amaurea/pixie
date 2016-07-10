import numpy as np, argparse, pixie, h5py, enlib.wcs
from enlib import enmap, utils, interpol, cg as encg, fft
parser = argparse.ArgumentParser()
parser.add_argument("tod")
parser.add_argument("ofile")
args = parser.parse_args()

bsize  = 480
nblock =   4
ndelay = bsize*nblock
nspin  = 8
fmax   = 7.4e12
nfreq  = 512

tod = pixie.read_tod(args.tod)
d   = tod.signal[0]
d   = d.reshape(-1,nspin,ndelay)
delay = tod.elements[4].reshape(-1,nspin,ndelay)

hfile = h5py.File(args.ofile,"w")
hfile["d"] = d

def fix_fourier(d):
	# Adjust phase to compensate for sky motion during a spin
	nt,nspin,ndelay = d.shape
	d = pixie.froll(d.reshape(-1,nspin*ndelay), np.arange(nspin*ndelay)/float(nspin*ndelay), 0).reshape(-1,nspin,ndelay)
	## Adjust phase to compensate for spin during strokes
	d  = pixie.froll(d, np.arange(ndelay)[None,None,:]/float(ndelay),1)
	return d

d  = fix_fourier(d)
hfile["d2"] = d

# Ok, at this point the dimensions are independent. The FT of the
# delay dimension is a projection of the spectrum. The spin-2 components
# of the spin direction describes Q and U. But what about T? Double
# barrel operation is normally not sensitive to T (except the DC component).
# The DC component just becomes an overall offset of the spectrum, which we
# get rid of by demanding spec[0]=0 or spec[-1]=0

# Build our delay geometry
dstep = delay[0,0,1]-delay[0,0,0]
dwcs = enlib.wcs.WCS(naxis=1)
dwcs.wcs.ctype[0] = 'TIME'
dwcs.wcs.cdelt[0] = dstep
dwcs.wcs.crpix[0] = 1

# Split into left-going and right-going scan and positive and negative frequencies
#d = d.reshape(-1,nspin,nblock,bsize)
#delay = delay.reshape(-1,nspin,nblock,bsize)
#
### Reorder all spectrograms to standard 0-high delay order
#for i in range(1,nblock,2):
#	d[:,:,i]     = d[:,:,i,::-1]
#	delay[:,:,i] = delay[:,:,i,::-1]
#d = d.reshape(-1,nspin,nblock*bsize)
hfile["d3"] = d
hfile["delay"] = delay

def stroke2spec(arr, wcs):
	ndelay = arr.shape[-1]
	spec   = fft.rfft(arr,axes=[-1]).real[...,::2]*2/ndelay
	owcs   = pixie.wcs_delay2spec(wcs, ndelay/4)
	return spec, owcs

# Try to recover first spectrum
#spec, swcs = pixie.delay2spec(d, dwcs, axis=-1)
spec, swcs = stroke2spec(d, dwcs)
freq = swcs.wcs_pix2world(np.arange(spec.shape[-1]),0)[0]
hfile["spec"] = spec
hfile["freq"] = freq

# Fourier-decompose spin
fd = fft.rfft(d, axes=[1])
dcomp = np.array([fd[:,0].real,fd[:,2].real,fd[:,2].imag])

hfile["dcomp"] = dcomp
# And spectra from this
#scomp, _ = pixie.delay2spec(dcomp, dwcs, axis=-1)
scomp, _ = stroke2spec(dcomp, dwcs)
hfile["scomp"] = scomp


hfile.close()
