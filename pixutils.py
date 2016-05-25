import numpy as np, warnings, astropy.io.fits, enlib.wcs
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
	"""Converts a spectrum cube arr[nfreq,...] into an
	autocorrelation function [delay,...]. The frequency
	information is described by the spectral wcs argument."""
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
	owcs.wcs.ctype[0] = 'TIME'
	# Take into account the units on the x-axis
	arr *= wcs.wcs.cdelt[0]
	return arr, owcs

def delay2spec(arr, wcs, axis=0, inplace=False, bsize=32):
	"""Converts an autocorrelation cube arr[ndelay,...] into an
	autocorrelation function [delay,...]. The delay
	information is described by the temporal wcs argument."""
	arr = np.asanyarray(arr)
	if not inplace: arr = arr.copy()
	with utils.flatview(arr, [axis], "rw") as aflat:
		n = aflat.shape[0]
		ndelay = aflat.shape[1]
		nb = (n+bsize-1)/bsize
		for bi in range(nb):
			i1, i2 = bi*bsize, min(n,(bi+1)*bsize)
			aflat[i1:i2] = fft.redft00(aflat[i1:i2]*2, normalize=True)
	# Update our wcs
	owcs = wcs.deepcopy()
	owcs.wcs.cdelt[0] = c/wcs.wcs.cdelt[0]/ndelay/2
	owcs.wcs.ctype[0] = 'FREQ'
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

def fullsky_geometry(res=0.1*utils.degree, dims=()):
	"""Build an enmap covering the full sky, with the outermost pixel centers
	at the poles and wrap-around points."""
	nx,ny = int(2*np.pi/res), int(np.pi/res)
	wcs   = enlib.wcs.WCS(naxis=2)
	wcs.wcs.crval = [0,0]
	wcs.wcs.cdelt = [360./nx,180./ny]
	wcs.wcs.crpix = [nx/2+1,ny/2+1]
	wcs.wcs.ctype = ["RA---CAR","DEC--CAR"]
	return dims+(ny+1,nx+1), wcs

def freq_geometry(fmax, nfreq):
	wcs = enlib.wcs.WCS(naxis=1)
	wcs.wcs.cdelt[0] = fmax/nfreq
	wcs.wcs.crpix[0] = 1
	wcs.wcs.ctype[0] = 'FREQ'
	return wcs

def longitude_geometry(box, res=0.1*utils.degree, dims=()):
	"""Produce a longitudinal geometry, which follows a stripe around a line
	of longitude, and is approximately flat in this strip. box is [{from,to},{dec,ra}],
	but the ra bounds only apply along dec=0 - elsewhere they will be larger."""
	# WCS works in degrees
	box = np.array(box)/utils.degree
	res = res/utils.degree
	# Find our center point. All our coordinates should be within
	# 180 degrees from this point, but our reference point can
	# only be along the equator. We can achieve this by putting
	# it at [ra0,0], assuming our patch isn't wider than 360 degrees,
	# which it shouldn't be.
	dec0, ra0 = np.mean(box,0)
	wdec, wra = box[1]-box[0]
	wcs = enlib.wcs.WCS(naxis=2)
	wcs.wcs.ctype = ['RA---CAR','DEC--CAR']
	wcs.wcs.cdelt = [res*np.sign(wdec),res*np.sign(wra)]
	wcs.wcs.crval = [ra0,0]
	wcs.wcs.crpix = [-(dec0-abs(wdec)/2.)/res+1,abs(wra)/2./res+1]
	wcs.wcs.lonpole = 90
	wcs.wcs.latpole = 0
	nra  = int(round(abs(wra/res)))
	ndec = int(round(abs(wdec/res)))
	return dims + (nra+1,ndec+1), wcs

def longitude_band_bounds(point, step=1, niter=10):
	"""point[{phi,theta],...], returns [{from,to},{theta,phi}], as that's what
	longitude geometry wants."""
	point = np.array(point).reshape(2,-1)[:,::step]/utils.degree
	# First find the minimum abs theta. That side of the sky will be our reference.
	imin  = np.argmin(np.abs(point[1]))
	phiref, thetaref = point[:,imin]
	# Find first point that's twice as close to one of the poles.
	# We will use this to define the reference pole
	pdist  = 90-np.abs(point[1])
	closer = np.where(pdist < pdist[imin]/2)[0]
	# If there is no such point, just use the one with the highest value
	imid = closer[0] if len(closer) > 0 else np.argmin(pdist)
	theta_closer = point[1,imid]
	# Our first esimate of the center phi is inaccurate,
	# so iterate to get a better mean.
	for i in range(niter):
		# Ok, define our wcs
		wcs = enlib.wcs.WCS(naxis=2)
		wcs.wcs.ctype = ['RA---CAR','DEC--CAR']
		wcs.wcs.cdelt = [1,1]
		wcs.wcs.crval = [phiref,0]
		wcs.wcs.crpix = [-thetaref+1,1]
		wcs.wcs.latpole = 0
		if theta_closer > 0:
			wcs.wcs.lonpole =  90
		else:
			wcs.wcs.lonpole = -90
		# Transform the points. Since cdelt = 1, these new
		# pixel coordinates will correspond to flat-sky angles
		x, y = wcs.wcs_world2pix(point[0], point[1], 0)
		box = np.array([
			[thetaref+np.min(x),phiref-np.max(y)],
			[thetaref+np.max(x),phiref-np.min(y)]])
		phiref -= np.mean(y)
	return box*utils.degree

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

def sim_reference_blackbody(shape, wcs, T=None):
	if T is None: T = Tcmb
	res = enmap.zeros(shape, wcs)
	res[0] = T
	return res

def read_healpix(fname):
	import healpy
	try:
		return np.array(healpy.read_map(fname, field=range(3)))
	except IndexError:
		return np.array(healpy.read_map(fname))

def project_healpix(shape, wcs, healmap, rot=None, verbose=False):
	import healpy
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
