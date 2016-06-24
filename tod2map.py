import numpy as np, argparse, pixie, h5py, enlib.wcs
from enlib import enmap, utils, interpol, cg as encg, fft
parser = argparse.ArgumentParser()
parser.add_argument("tod")
parser.add_argument("odir")
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

hfile = h5py.File("test.hdf","w")
hfile["d"] = d

# FIXME: Should ensure periodicity at this point.

def fix_fourier(d):
	# Adjust phase to compensate for sky motion during a spin
	nt,nspin,ndelay = d.shape
	d = pixie.froll(d.reshape(-1,nspin*ndelay), np.arange(nspin*ndelay)/float(nspin*ndelay), 0).reshape(-1,nspin,ndelay)
	## Adjust phase to compensate for spin during strokes
	d  = pixie.froll(d, np.arange(ndelay)[None,None,:]/float(ndelay),1)
	return d

def fix_fourier2(d):
	# Adjust phase to compensate for sky motion during a spin
	nt,nspin,ndelay = d.shape
	# Here we roll along the time axis for each [spin,delay]-cell. This is periodic because
	# d[0+nt,sub] = d[nt*nsub+sub] = d[N+sub] = d[sub] = d[0,sub]
	d = pixie.froll(d.reshape(-1,nspin*ndelay), np.arange(nspin*ndelay)/float(nspin*ndelay), 0).reshape(-1,nspin,ndelay)
	# Here we roll along the spin axis for each [t,delay].
	# d[t,0+nspin,delay] = d[t*nspin*ndelay + nspin*ndelay + delay] =
	# d[(t+1)*nspin*ndelay + delay] = d[t+1,0,delay]
	# If all we had were the underlying guarantee that d as a whole is periodic, then
	# a shift in the spin direction is non-periodic. But since we have already corrected
	# for the drift, each spin cycle *should* be periodic after all. So padding would just
	# make things worse.
	dnext = np.roll(d, 1,0)
	dprev = np.roll(d,-1,0)
	dpad = np.concatenate([d,dnext,dnext[::-1],d[::-1],dprev[::-1],dprev],1)
	dpad = pixie.froll(dpad, np.arange(ndelay)[None,None,:]/float(ndelay),1)
	d    = dpad[:,:nspin]
	return d

def fix_bicubic(d):
	# Position of unshifted array into shifted array
	nt,nspin,ndelay = d.shape
	inds = np.mgrid[:nt,:nspin,:ndelay].astype(float)
	inds[0] += dir*(inds[1]+inds[2]/ndelay)/nspin
	inds[1] += dir*inds[2]/ndelay
	return interpol.map_coordinates(d, inds, order=3, border="cyclic").reshape(-1)

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
dwcs.wcs.crpix[0] = dstep

# Split into left-going and right-going scan and positive and negative frequencies
d = d.reshape(-1,nspin,nblock,bsize)
delay = delay.reshape(-1,nspin,nblock,bsize)

# Reorder all spectrograms to standard 0-high delay order
for i in range(1,nblock,2):
	d[:,:,i]     = d[:,:,i,::-1]
	delay[:,:,i] = delay[:,:,i,::-1]
hfile["d3"] = d
hfile["delay"] = delay

# Try to recover first spectrum
spec, swcs = pixie.delay2spec(d, dwcs, axis=-1)
freq = swcs.wcs_pix2world(np.arange(spec.shape[-1]),0)[0]
hfile["spec"] = spec
hfile["freq"] = freq

# Fourier-decompose spin
fd = fft.rfft(d, axes=[1])
dcomp = np.array([fd[:,0].real,fd[:,2].real,fd[:,2].imag])

hfile["dcomp"] = dcomp
# And spectra from this
scomp, _ = pixie.delay2spec(dcomp, dwcs, axis=-1)
hfile["scomp"] = scomp


hfile.close()
