"""This module implements a spectrogram generator. It takes in
input maps with descriptors of their spectra, and lets you query
it for simulated spectrograms."""
import numpy as np, pixutils
from enlib import utils, enmap

# 1. The input maps may not have the same beam. We therefore need
#    to apply beams per component.
# 2. If the generator is queried independently per beam, then
#    the work involved in interpolating the values and evaluating
#    the spectrum is duplicated.
#
# Here's how I would want things to work:
# comps, cbeams = generate_patch(patch desc)
# for each beam_profile:
#   spec_map = 0
#   for each comp, cbeam:
#     spec_map += switch_beam(comp, beam_profile, cbeam)
#   delay_map[beam_profile] = spec2delay(spec_map)
# Generate the TQU incident at the bottom of each barrel, in
# detector coordinates.
# for each obarrel:
#   for each ibarrel:
#      How TQU at the bottom (output side) of barrel obarrel
#      relates to the TQU incident at the top (input side) of
#      barrel ibarrel. [{dc,delay},{TQU},{TQU}] per sample.
#      resp_params = calc_resp_params(gamma, ibarrel, obarrel)
#      for beam_profile, beam_resp in beam_decompositions[ibarrel]:

# Should not calculate pointing per barrel. Should do so per
# beam component. In a nice case these would coincide, but
# for more complicated beams we want the beam offsets in
# focaplane coordinates, and must therefore apply them before
# the coordinate transformation.
#
# Should have a different name for the input and output barrels.
# How about "barrel" for the sky-facing part and "horn" for the
# detector-facing part?
#
# So perhaps our whole program loop would be:
#
# for each sample chunk:
#    orbit = generate_orbit(chunk)
#    for each sky:
#       shape, wcs  = estimate_patch_bounds(orbit)
#       comp_info   = get_patch_components(sky, shape, wcs)
#       for each beam_type:
#          patches[sky][beam_type] = get_patch_spectrogram(comp_info, beam_type)
#    for each barrel:
#       horn_sig = 0
#       for each subbeam in barrel:
#          point = calc_pointing(orbit, subbeam)
#          resp  = calc_response(point, barrel) # [nhorn,{dc,delay},...]
#          horn_sig += calc_barrel_signal(patches[barrel.sky][subbeam.type], resp)
#       for each det in all barrels:
#          det_sig[det] += calc_det_signal(horn_sig[det.horn], det.response)
#    det_sig = downsample(det_sig)

# That's a pretty nice structure. Let's aim for a program whose
# main function is as simple as this.
#
# field:     .name, .map, .beam, .spec
# sky:       .name, .fields
# comp_info: [(.name, .specmap, .beam),...]
# barrel:    .id, .sky, .beam
# beam:      .subbeams
# subbeam:   .beam_type, .offsets, .response
# detector:  .horn, .response

# Actually, how does per detector beams work in this setup?
# Barrel beams are easy because they happen before all the
# interferometry - each barrel just sees a differnt sky.
# But for detector beams, the difference would come from
# stuff inside the interferometer, like the filters or
# the mirror. Which means that their effect would be
# spread between multiple skies. That's nasty. Let's
# not do that for now.

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
