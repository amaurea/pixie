# for each sample chunk:
#    orbit = generate_orbit(chunk)
#    for each sky:
#       shape, wcs  = estimate_patch_bounds(orbit)
#       for each field in sky:
#          subfield = field.project(shape, wcs)
#          for each beam_type:
#             patches[sky][beam_type] += get_patch_spectrogram(subfield, beam_type)
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
#
# spec is a function (freq,map) -> map[nfreq,...]
# beam_component is a function (freq,l) -> val[{fshape},{lshape}]
# beam_type is just an index into a list of beam_components

def read_field(fname):
	"""Read a field from a specially prepared fits file."""
	map    = enmap.read_map(fname)
	header = astropy.io.fits.open(fname)[0].header
	name   = header["NAME"]
	# Get the spectral information and build a Spectrum object
	spec_type = header["SPEC"]
	if spec_type == "BLACK":
		spec = SpecBlackbody(name)
	elif spec_type == "GRAY":
		spec = SpecGraybody(name, T=float(header["TBODY"]), beta=float(header["BETA"]),
				fref=float(header["FREF"]), unit=float(header["SUNIT"]))
	else: raise ValueError("Spectrum type '%s' not recognized" % spec_type)
	# Get the beam information
	beam_type = header["BEAM"]
	if beam_type == "NONE":
		beam = BeamNone()
	elif beam_type == "GAUSS":
		beam = BeamGauss(float(header["FWHM"])*utils.degree)
	else: raise ValueError("Beam type '%s' not recognized" % beam_type)
	return Field(name.lower(), map, spec, beam)

class Field:
	def __init__(self, name, map, spec, beam, order=3):
		"""A field comprising one component of the sky. Specified
		by an enmap "map", a spectrum function spec, and a beam
		function beam"""
		self.name = name
		self.spec = spec
		self.beam = beam
		self.order= order
		self.map  = utils.interpol_prefilter(map, order=order)
	def project(self, shape, wcs):
		"""Project our values onto a patch with the given shape and wcs. Returns
		a new Field."""
		pos = enmap.posmap(shape, wcs)
		map = self.map.at(pos, order=self.order, prefilter=False)
		return Field(self.name, map, self.spec, self.beam, self.order)

##### Spectrum types #####

class SpecBlackbody:
	type = "blackbody"
	def __init__(self, name):
		self.name = name
	def __call__(self, freq, map):
		res = enmap.samewcs(np.zeros(freq.shape + map.shape, map.dtype), map)
		res[:,0]  = pixutils.blackbody(freq, map[0])
		# Constant polarization fraction by frequency
		res[:,1:] = res[:,0,None] * (map[None,1:]/map[None,None,0])
		return res

class SpecGraybody:
	type = "graybody"
	def __init__(self, name, T, beta, fref, unit):
		self.name = name
		self.T    = T
		self.beta = beta
		self.fref = fref
		self.unit = unit
	def __call__(self, freq, map):
		res   = enmap.samewcs(np.zeros(freq.shape + map.shape, map.dtype), map)
		scale = pixutils.graybody(freq, self.T, self.beta) / pixutils.graybody(self.fref, self.T, self.beta) * self.unit
		res  += scale[:,None,None,None] * map[None,:,:,:]
		return res

##### Beam types #####

class BeamNone:
	type = "none"
	def __call__(self, freq, l):
		return np.full(freq.shape + l.shape, 1.0)

class BeamGauss:
	type = "gauss"
	def __init__(self, fwhm):
		self.fwhm = fwhm
	def __call__(self, freq, l):
		res = np.zeros(freq.shape, + l.shape)
		res[:] = np.exp(-0.5*l**2*(self.fwhm**2/(8*np.log(2))))
		return res
