import numpy as np, argparse, pixie, h5py, enlib.wcs, os, time
import astropy.io.fits as fitsio
from enlib import enmap, utils, mpi
parser = argparse.ArgumentParser()
parser.add_argument("rings", nargs="+")
parser.add_argument("ofile")
parser.add_argument("-d", "--dets", type=str, default="")
parser.add_argument("-C", "--config", type=str, default=None)
parser.add_argument("-I", "--individual", action="store_true")
args = parser.parse_args()

config= pixie.load_config(args.config)
comm  = mpi.COMM_WORLD

# Read the first ring to get the coordinate system
hdu   = fitsio.open(args.rings[0])[0]
wcs   = enlib.wcs.WCS(hdu.header)
shape = hdu.data.shape
dtype = np.float64
atpole= True

ndet, nfreq, ncomp, ntheta, nphi = shape
nphi    = int(np.ceil(360 / wcs.wcs.cdelt[0]))
dets = np.arange(ndet)
ndet = eval("dets" + args.dets).size

# Rings cover 360 degrees, but those are split into two
# columns in our image. The poles are part of both sides.
ntheta = ntheta/2 + atpole

# We want our output coordinate system to be centered on the equator
# rather than beginning there. So offset crpix for theta
owcs = wcs.deepcopy().sub(4)
owcs.wcs.crpix[0] += nphi/2
owcs.wcs.crpix[1] += ntheta/2
owcs.wcs.ctype = ["RA---CAR", "DEC--CAR", "STOKES", "FREQ"]
oshape = [ndet,nfreq,ncomp,ntheta,nphi]

# Ok, create our output map
omap = enmap.zeros(oshape, owcs.sub(2), dtype=dtype)
hits = enmap.zeros([ntheta,nphi])

# Loop through all our rings, and copy over the data
t0 = time.time()
for ifile in args.rings[comm.rank::comm.size]:
	print "%2d %6.3f %s" % (comm.rank, time.time()-t0, ifile)
	# Read all detectors
	m = enmap.read_map(ifile)
	m = eval("m" + args.dets)
	# Find the coordinates of each pixel in the ring
	nring = m.shape[-2]
	theta,  phi   = m.pix2sky([0,0])
	itheta, iphi  = np.round(omap.sky2pix([theta,phi])).astype(int)
	itheta = itheta + np.arange(nring)
	iphi   = iphi   + np.arange(nring)*0
	# Wrap onto sky
	bad = itheta >= ntheta
	itheta[bad] = 2*ntheta - itheta[bad] - 1 - atpole
	iphi  [bad] = iphi[bad] + nphi/2
	bad = itheta < 0
	itheta[bad] = -itheta[bad] -1 + atpole
	iphi  [bad] = iphi[bad] + nphi/2
	iphi %= nphi
	#theta = theta + np.arange(nring)*360./nring
	#for i in range(nring):
	#	print "%4d %7.3f %4d %4d" % (i, theta[i], itheta[i], ntheta)
	#1/0
	omap[...,itheta,iphi] += m[...,0]
	hits[itheta,iphi] += 1
	# Copy over poles
	if atpole:
		for t in [0,ntheta-1]:
			i = np.where(itheta == t)[0]
			omap[...,itheta[i],(iphi[i]+nphi/2)%nphi] += m[...,i,0]
			hits[itheta[i],(iphi[i]+nphi/2)%nphi] += 1

omap = utils.allreduce(omap, comm)
hits = utils.allreduce(hits, comm)
omap /= np.maximum(1,hits)
omap.wcs = owcs

if not args.individual:
	omap = np.mean(omap,0)

if comm.rank == 0:
	enmap.write_map(args.ofile, omap)
