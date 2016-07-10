import numpy as np, argparse, pixie, h5py, enlib.wcs, os
import astropy.io.fits as fitsio
from enlib import enmap, utils, mpi
parser = argparse.ArgumentParser()
parser.add_argument("rings", nargs="+")
parser.add_argument("ofile")
parser.add_argument("-d", "--det", type=int, default=0)
args = parser.parse_args()

comm = mpi.COMM_WORLD

# Read the first ring to get the coordinate system
hdu   = fitsio.open(args.rings[0])[0]
wcs   = enlib.wcs.WCS(hdu.header)
shape = hdu.data.shape
dtype = np.float32

ndet, nfreq, ncomp, ntheta, nphi = shape
nphi    = int(np.ceil(360 / wcs.wcs.cdelt[0]))
# Rings cover 360 degrees, but those are split into two
# columns in our image.
ntheta /= 2

# We want our output coordinate system to be centered on the equator
# rather than beginning there. So offset crpix for theta
owcs = wcs.deepcopy().sub(4)
owcs.wcs.crpix[0] += nphi/2
owcs.wcs.crpix[1] += ntheta/2
owcs.wcs.ctype = ["RA---CAR", "DEC--CAR", "STOKES", "FREQ"]
oshape = [nfreq,ncomp,ntheta,nphi]

# Ok, create our output map
omap = enmap.zeros(oshape, owcs.sub(2), dtype=dtype)
hits = omap[0,0]*0

# Loop through all our rings, and copy over the data
for ifile in args.rings[comm.rank::comm.size]:
	print ifile
	m = enmap.read_map(ifile)[args.det]
	nring = m.shape[-2]
	theta,  phi   = m.pix2sky([0,0])
	itheta, iphi  = omap.sky2pix([theta,phi]).astype(int)
	itheta = itheta + np.arange(nring)
	iphi   = iphi   + np.arange(nring)*0
	# Wrap onto sky
	bad = itheta >= ntheta
	itheta[bad] = 2*ntheta - itheta[bad] - 1
	iphi  [bad] = iphi[bad] + nphi/2
	bad = itheta < 0
	itheta[bad] = -itheta[bad]
	iphi  [bad] = iphi[bad] + nphi/2
	iphi %= nphi
	omap[:,:,itheta,iphi] = m[:,:,:,0]
	hits[itheta,iphi] += 1

omap = utils.allreduce(omap, comm)
hits = utils.allreduce(hits, comm)
omap /= np.maximum(1,hits)
omap.wcs = owcs

if comm.rank == 0:
	enmap.write_map(args.ofile, omap)
