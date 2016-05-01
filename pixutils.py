import numpy as np, warnings, astropy.io.fits, enlib.wcs
from enlib import enmap, fft, utils

h  = 6.626070040e-34
c  = 299792458
kb = 1.38064853e-23

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
	# FIXME: Check units on this. It's also pretty slow.
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
