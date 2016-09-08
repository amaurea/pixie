import numpy as np, argparse, pixie, h5py, enlib.wcs, os
from enlib import enmap, utils, fft, mpi, bunch
parser = argparse.ArgumentParser()
parser.add_argument("tods", nargs="+")
parser.add_argument("odir")
parser.add_argument("-C", "--config", type=str, default=None)
args = parser.parse_args()

comm = mpi.COMM_WORLD
fft.engine = "fftw"

# We assume that each input tod corresponds to a single ring.
# For each tod we will produce the map along the ring it
# corresponds to, and write it to a ring file in the given output
# directory. The ring will have pixels of size 360 deg * spin_period
# / orbit_period. In the end we will have variable stroke amplitude,
# but for now we will assume it's constant at the level in config.

config = pixie.load_config(args.config)
ncomp  = 3
ncopy  = 4
ndelay = int(round(config.delay_period / config.sample_period))
nspin  = int(round(config.spin_period  / config.delay_period))
ntheta = int(round(config.scan_period  / config.spin_period))
nphi   = int(round(config.orbit_period / config.scan_period))
nfreq  = ndelay / ncopy
nsamp  = ntheta * nspin * ndelay

# Our largest delay is config.delay_amp. The lowest frequency
# is c/delay_amp/2 because the lowest frequency is the one
# where we have half a period in our interval.
dfreq  = utils.c / config.delay_amp / 2

# Get a representative filter. Assume this filter is used for
# all tods and detectors.
filter  = pixie.parse_filter(config.filters[config.fiducial_filter])
dets    = [pixie.parse_det(det) for det in config.dets]

# Set up the ring WCS, which will be an unwrapped version
# of the final map's wcs. We will override crpix phi for
# each ring.
wcs = enlib.wcs.WCS(naxis=4)
wcs.wcs.ctype = ["PHI", "THETA", "STOKES", "FREQ"]
wcs.wcs.cdelt = [360.0/nphi, 360.0/ntheta, 1, dfreq]
wcs.wcs.crval = [0, 0, 0, 0]
wcs.wcs.crpix = [1, 1, 1, 1]

utils.mkdir(args.odir)

def dump(pre, d):
	with h5py.File(pre + ".hdf", "w") as hfile:
		hfile["data"] = d

def polar(vec): return np.sum(vec**2)**0.5, np.arctan2(vec[1],vec[0])
def change_response(resp_old, resp_new, data):
	resp_old = np.array(resp_old,dtype=float)
	resp_new = np.array(resp_new,dtype=float)
	res = np.array(data)
	# T is simple. Just rescale
	res[:,0] *= resp_new[0]/resp_old[0]
	# P is a rotation plus a rescaling
	amp_old, ang_old = polar(resp_old[1:])
	amp_new, ang_new = polar(resp_new[1:])
	R = utils.rotmatrix(ang_new-ang_old, 'z')[:2,:2] * amp_new / amp_old
	res[:,1:] = np.einsum("ab,fb...->fa...",R,res[:,1:])
	return res
def change_horn(horn_old, horn_new, data):
	res = np.array(data)
	if horn_old != horn_new: res[:,0] *= -1
	return res

for fname in args.tods[comm.rank::comm.size]:
	print fname
	# Read the tod. tod.signal has units W/sr/m^2.
	tod = pixie.read_tod(fname, nsamp=nsamp)
	pre = args.odir + "/" + os.path.basename(fname)[:-4]
	# Center our coordinate system on our column. We didn't save
	# the pointing, so we have to compute it from the elements
	e     = tod.elements[:,:10]
	elem  = bunch.Bunch(ctime=e[0], orbit=e[1], scan=e[2], spin=e[3], delay=e[4])
	pgen  = pixie.PointingGenerator(**config.__dict__)
	orient= pgen.calc_orientation(elem)
	point = pgen.calc_pointing(orient, elem.delay, np.array([0,0,0]))
	phi   = point.angpos[0,0]
	wcs.wcs.crpix[0] = 1 - phi/utils.degree/wcs.wcs.cdelt[0]
	# Deconvolve time filter. This has almost no effect
	readout_sim = pixie.ReadoutSim(config)
	tod.signal = readout_sim.apply_filters(tod.signal, exp=-1)
	# Detector measures 1/4 of the signal due to how the interferrometry
	# is set up.
	gain  = 1.0/4
	# Get out tod samples
	d   = tod.signal / gain
	dump(pre+"_1", d)
	# Deconvolve the sample window. Each of our samples
	# is approximately the integral of the signal inside its
	# duration.
	fd  = fft.rfft(d, axes=[-1])
	fd /= np.sinc(np.arange(fd.shape[-1],dtype=float)/d.shape[-1])
	d   = fft.ifft(fd, d, axes=[-1], normalize=True)
	# Undo the effect of drift in theta and spin during each stroke
	d   = d.reshape(d.shape[0], ntheta, nspin, ndelay)
	d   = pixie.fix_drift(d)
	dump(pre+"_2", d)
	# Fourier-decompose the spin. *2 for pol because <sin^2> = 0.5.
	fd  = fft.rfft(d, axes=[2])/d.shape[2]
	d   = np.array([fd[:,:,0].real, fd[:,:,2].real*2, fd[:,:,2].imag*2])
	dump(pre+"_3", d)
	# Go from stroke to spectrum. This takes us to W/sr/m^2/Hz
	d   = fft.rfft(d).real[...,::2]*2/ndelay/dfreq
	dump(pre+"_4", d)
	# Unapply the frequency filter
	freqs = np.arange(d.shape[-1])*dfreq
	d  /= filter(freqs)
	dump(pre+"_5", d)
	# Reorder from [stokes,det,theta,phi,freq] to [det,freq,stokes,theta,phi]
	d   = d[:,:,:,None,:]
	d   = utils.moveaxes(d, (1,4,0,2,3), (0,1,2,3,4))
	# Apply the detector responses.
	for di, det in enumerate(dets):
		d[di] = change_response(det.response, [1,1,0], d[di])
		d[di] = change_horn(det.horn, 0, d[di])
	# Output as a ring file
	m   = enmap.enmap(d, wcs, copy=False)
	enmap.write_map(pre + ".fits", m)
