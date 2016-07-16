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

# Our largest delay is config.delay_amp. The lowest frequency
# is c/delay_amp/2 because the lowest frequency is the one
# where we have half a period in our interval.
dfreq  = utils.c / config.delay_amp / 2

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

for fname in args.tods[comm.rank::comm.size]:
	print fname
	tod = pixie.read_tod(fname)
	pre = args.odir + "/" + os.path.basename(fname)[:-4]
	# Center our coordinate system on our column. We didn't save
	# the pointing, so we have to compute it from the elements
	e     = tod.elements[:,:10]
	elem  = bunch.Bunch(ctime=e[0], orbit=e[1], scan=e[2], spin=e[3], delay=e[4])
	pgen  = pixie.PointingGenerator(**config.__dict__)
	orient= pgen.calc_orientation(elem)
	point = pgen.calc_pointing(orient, elem.delay, np.array([0,0,0]))
	phi   = point.angpos[0,0]
	gamma0= point.gamma[0]
	wcs.wcs.crpix[0] = 1 - phi/utils.degree/wcs.wcs.cdelt[0]
	# Ger out tod samples
	d   = tod.signal
	d   = d.reshape(d.shape[0], ntheta, nspin, ndelay)
	# Undo the effect of drift in theta and spin during each stroke
	d   = pixie.fix_drift(d)
	# Fourier-decompose the spin. The details of this should depend on the
	# barrel and detector configuration. Signs here are weird.
	fd  = fft.rfft(d, axes=[2])
	d   = np.array([-fd[:,:,0].real, fd[:,:,2].real, -fd[:,:,2].imag])
	# Rotate polarization. Why is this needed?  Why is it spin 4?
	d   = pixie.rot_comps(d, -gamma0*4, 0)
	# Go from stroke to spectrum
	d   = fft.rfft(d).real[...,::2]*2/ndelay
	# Reorder from [stokes,det,theta,phi,freq] to [det,freq,stokes,theta,phi]
	d   = d[:,:,:,None,:]
	d   = utils.moveaxes(d, (1,4,0,2,3), (0,1,2,3,4))
	# TODO: combine detectors
	# Output as a ring file
	m   = enmap.enmap(d, wcs, copy=False)
	enmap.write_map(pre + ".fits", m)
	dump(pre, d)
