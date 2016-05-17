import numpy as np, warnings, astropy.io.fits, enlib.wcs, healpy
from enlib import enmap, fft, utils, curvedsky, lensing, aberration, sharp, coordinates

Tcmb= 2.725
h   = 6.626070040e-34
c   = 299792458
kb  = 1.38064853e-23

def broadcast_stack(a, b):
	a = np.asanyarray(a)
	b = np.asanyarray(b)
	adim, bdim = a.ndim, b.ndim
	a = a[(Ellipsis,) + (None,)*bdim]
	b = b[(None,)*adim]
	return a, b

def blackbody(freqs, T, deriv=False):
	"""Given a set of frequencies freqs[{fdims}] and a set of
	temperatures T[{tdims}], returns a blackbody spectrum
	spec[{fdims},{tdims}]. All quantities are in SI units."""
	return graybody(freqs, T, 0, deriv=deriv)

def graybody(freqs, T, beta, deriv=False):
	"""Given a set of frequencies freqs[{fdims}] and a set of
	temperatures T[{tdims}] and emmisivities beta[{tdims}], returns a graybody spectrum
	spec[{fdims},{tdims}]. All quantities are in SI units."""
	freqs, T = broadcast_stack(freqs, T)
	beta = np.asanyarray(beta)
	with warnings.catch_warnings():
		warnings.filterwarnings("ignore")
		ex = np.exp(h*freqs/(kb*T))
		res = (2*h/c**2) * freqs**(3+beta) / (ex - 1)
		if deriv:
			res *= ex/(ex-1)*(h/kb)*freqs/T**2
		if res.ndim == 0:
			if ~np.isfinite(res): res = 0
		else:
			res[~np.isfinite(res)] = 0
		return res

def spec2delay(arr, wcs, axis=0, inplace=False, bsize=32):
	arr = np.asanyarray(arr)
	if not inplace: arr = arr.copy()
	with utils.flatview(arr, [axis], "rw") as aflat:
		n = aflat.shape[0]
		nfreq = aflat.shape[1]
		nb = (n+bsize-1)/bsize
		for bi in range(nb):
			i1, i2 = bi*bsize, min(n,(bi+1)*bsize)
			aflat[i1:i2] = fft.redft00(aflat[i1:i2]) * 0.5
	owcs = wcs.deepcopy()
	owcs.wcs.cdelt[0] = c/wcs.wcs.cdelt[0]/nfreq/2
	return arr, owcs

def read_map(fname, hdu=0):
	m = enmap.read_map(fname)
	hdu = astropy.io.fits.open(fname)[hdu]
	spec_wcs = enlib.wcs.WCS(hdu.header).sub([4])
	return m, spec_wcs

def write_map(fname, m, spec_wcs):
	m = m.copy()
	w = m.wcs.sub([1,2,0,0])
	w.wcs.cdelt[3] = spec_wcs.wcs.cdelt[0]
	w.wcs.crpix[3] = spec_wcs.wcs.crpix[0]
	w.wcs.ctype[3] = spec_wcs.wcs.ctype[0]
	header = w.to_header()
	header['NAXIS'] = m.ndim
	for i,n in enumerate(m.shape[::-1]):
		header['NAXIS%d'%(i+1)] = n
	hdus  = astropy.io.fits.HDUList([astropy.io.fits.PrimaryHDU(m, header)])
	with warnings.catch_warnings():
		warnings.filterwarnings('ignore')
		hdus.writeto(fname, clobber=True)

def arr2fullsky(arr):
	arr = np.asarray(arr)
	wcs = enlib.wcs.WCS(naxis=2)
	wcs.wcs.crval = [0,0]
	wcs.wcs.cdelt = [360,180]
	wcs.wcs.crpix = [1.5,1.5]
	wcs.wcs.ctype = ["RA---CAR","DEC--CAR"]
	oarr = arr[...,None,None] + np.zeros([2,2],dtype=arr.dtype)
	return enmap.enmap(oarr, wcs)

def build_fullsky_geometry(res=0.1, dims=()):
	"""Build an enmap covering the full sky, with the outermost pixel centers
	at the poles and wrap-around points."""
	nx,ny = int(360/res), int(180/res)
	wcs   = enlib.wcs.WCS(naxis=2)
	wcs.wcs.crval = [0,0]
	wcs.wcs.cdelt = [360./nx,180./ny]
	wcs.wcs.crpix = [nx/2+1,ny/2+1]
	wcs.wcs.ctype = ["RA---CAR","DEC--CAR"]
	return dims+(ny+1,nx+1), wcs

def sim_cmb_map(shape, wcs, powspec, ps_unit=1e-6, lmax=None, seed=None, T_monopole=None, verbose=False, beta=None, aberr_dir=None, oversample=2):
	"""Simulate lensed cmb map with the given [phi,T,E,B]-ordered
	spectrum, including the effects of sky curvature and aberration."""
	# First simulate a curved-sky lensed map.
	res = wcs.wcs.cdelt[0]
	if lmax is None: lmax = 2*int(180./res)
	m_unlensed, m_lensed = lensing.rand_map(shape, wcs, powspec, lmax=lmax, seed=seed, output="ul", verbose=verbose, oversample=oversample)
	# Correct unit to get K, and add CMB monopole
	if verbose: print "Adding monopole"
	if T_monopole is None: T_monopole = Tcmb
	m_unlensed *= ps_unit
	m_lensed   *= ps_unit
	m_unlensed[0] += T_monopole
	m_lensed[0]   += T_monopole
	# Compute the aberrated and modulated map
	if verbose: print "Aberrating"
	if beta is None: beta = aberration.beta
	if aberr_dir is None: aberr_dir = aberration.dir_ecl
	m_abber = aberration.aberrate(m_lensed, dir=aberr_dir, beta=beta)
	return m_unlensed, m_lensed, m_abber

def read_healpix(fname):
	try:
		return np.array(healpy.read_map(fname, field=range(3)))
	except IndexError:
		return np.array(healpy.read_map(fname))

def project_healpix(shape, wcs, healmap, rot=None, verbose=False):
	ncomp, npix = healmap.shape
	nside = healpy.npix2nside(npix)
	lmax  = 3*nside
	# Set up sharp, which we will use for the projection.
	# Could do the first part in healpix, but that would be slower.
	minfo = sharp.map_info_healpix(nside)
	ainfo = sharp.alm_info(lmax)
	sht   = sharp.sht(minfo, ainfo)
	alm   = np.zeros((ncomp,ainfo.nelem), dtype=np.complex)
	if verbose: print "healpix T to alm"
	sht.map2alm(healmap[0], alm[0])
	if ncomp == 3:
		if verbose: print "healpix P to alm"
		sht.map2alm(healmap[1:3], alm[1:3])
	del healmap
	# Compute the output locations
	if verbose: print "computing positions"
	pos = enmap.posmap(shape, wcs)
	if rot:
		if verbose: print "rotating positions"
		s1,s2 = rot.split(",")
		opos = coordinates.transform(s2, s1, pos[::-1], pol=ncomp==3)
		pos[...] = opos[1::-1]
		if len(opos) == 3: psi = -opos[2].copy()
		del opos
	# Can now project onto these locations. If we knew we wouldn't
	# be doing any rotation, this could be done faster by using
	# alm2map_cyl.
	if verbose: print "projecting on positions"
	res = curvedsky.alm2map_pos(alm, pos)
	# Take into account the rotation of the polarization basis
	if rot and ncomp==3:
		if verbose: print "rotating polarization"
		res[1:3] = enmap.rotate_pol(res[1:3], psi)
	return res
