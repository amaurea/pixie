"""This module implements a spectrogram generator. It takes in
input maps with descriptors of their spectra, and lets you query
it for simulated spectrograms."""
import numpy as np, pixutils
from enlib import utils, enmap

class SpectrogramGenerator:
	def __init__(self, inputs=[], order=3, unit=1e-20):
		self.order  = order
		self.unit   = unit
		# Prefilter our inputs, since we will be doing many lookups in them
		self.inputs = [
				(utils.interpol_prefilter(inp[0], inp[0].ndim-2, order=order), inp[1])
				for inp in inputs]
	def get_raw_spectrum_at(self, pos):
		"""Given pos[{freq,dec,ra}] evaluate the spectrum (not spectrogram)
		at these locations."""
		#pos = wrap_el(pos)
		res = None
		for i, (map,fun) in enumerate(self.inputs):
			# Fist evaluate the map at the given positions
			ivals = map.at(pos[1:], order=self.order, prefilter=False)
			ovals = fun(pos[0], ivals)
			ovals /= self.unit
			if res is None: res = ovals
			else: res += ovals
		return res
	def get_raw_specmap(self, shape, wcs, wcs_freq):
		"""Evaluate the spectra at the locations given by the
		spectral map corresponding to shape, wcs and wcs_freq."""
		pos = [wcs_freq.wcs_pix2world(np.arange(shape[-3]),0)[0]]
		pos += list(enmap.posmap(shape,wcs))
		return enmap.ndmap(self.get_raw_spectrum_at(pos), wcs), wcs_freq

def read_spec_input(fname_desc):
	toks = fname_desc.split(":")
	map = enmap.read_map(toks[0])
	fun = parse_spec_desc(toks[1:])
	return map, fun

def parse_spec_desc(toks):
	"""Given a string descriptor of the spectrum, parses it and returns
	a function(freq, map) -> [freq,[TQU],...], where map is a map that
	is associated with the descriptor."""
	spec_type = toks[0]
	if spec_type in ["black","blackbody"]:
		def specfun(freq, map):
			res = enmap.samewcs(np.zeros(freq.shape + map.shape, map.dtype), map)
			res[:,0]  = pixutils.blackbody(freq, map[0])
			# Constant polarization fraction by frequency
			res[:,1:] = res[:,0,None] * (map[None,1:]/map[None,None,0])
			return res
	elif spec_type in ["gray","graybody"]:
		Tbody = ftok(toks, 1, 19.6)
		beta  = ftok(toks, 2, 1.59)
		fref  = ftok(toks, 3, 600e9)
		unit  = ftok(toks, 4, 1e-20)
		def specfun(freq, map):
			res   = enmap.samewcs(np.zeros(freq.shape + map.shape, map.dtype), map)
			scale = pixutils.graybody(freq, Tbody, beta) / pixutils.graybody(fref, Tbody, beta) * unit
			res  += scale[:,None,None,None] * map[None,:,:,:]
			return res
	else:
		raise ValueError("Spectrum type '%s' not recognized" % spec_type)
	return specfun

def ftok(toks, i, default):
	return float(default) if len(toks) <= i else float(toks[i])

def wrap_el(pos):
	"""Implement wraparound in dec, such that values above pi/2
	wrap around to the opposite RA."""
	# First get rid of whole multiples of 2 pi, which do nothing
	pos = np.array(pos)
	pos[1] = utils.rewind(pos[1], 0, 2*np.pi)
	# This leaves us with the single-wrap stuff. nwrap will
	# be -1 for things that are below -pi/2, 0 for the standard range,
	# and +1 for values aboe pi/2.
	nwrap = np.int((pos[1]+np.pi/2)/(np.pi))
	odd   = nwrap % 2 == 1
	pos[1,odd] = np.pi*nwrap - pos[1,odd]
	pos[2,odd] += np.pi
	return pos
