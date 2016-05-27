# for each sample chunk:
#    orbit = generate_orbit(chunk)
#    brot  = calc_telescope_orientation(orbit)
#    for each sky:
#       shape, wcs  = estimate_patch_bounds(orbit)
#       for each field in sky:
#          subfield = field.project(shape, wcs)
#          for each beam_type:
#             specmap = calc_specmap(subfield, freqs, beam_type)
#             patches[sky][beam_type] += calc_spectrogram(specmap, freq_wcs)
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

# I want a pixie class at the top level, which is initialized using
# configuration object, and which can be called to generate tod
# chunks. But how should I do the pointing? Should pixie do it
# directly, or should I have a pointing generator class that
# pixie calls?
#
# * Many configuration parameters configure the pointing.
#   A pointing generator would either have to take a config
#   object itself, or have lots of arguments in the constructor.
# * There are three logical steps in the pointing generation:
#   orbital parameters, boresight rotmatrix and beam pointing.
#   The first two depend on several parameters, so would naturally
#   be methods of an object.
# * gen = PointingGenerator(...) / gen = Pixie(...)
#   orbit = gen.calc_orbit(chunk)
#   brot  = gen.calc_boresight(orbit)
#   point = gen.calc_pointing(orbit, brot, subbeam)
#   This approach means we need to pass around quite a few arguments
# * gen = ...
#   gen.calc_orbit(chunk)
#   gen.calc_boresight()
#   gen.calc_pointing(subbeam)
#   gen.orbit, gen.brot, gen.point
#   This version modifies gen instead. Shorter, but
#   I don't like the hidden variables.

##### Pointing #####

class PointingGenerator:
	"""This class handles the instrument's pointing, but does not care
	about details like sampling, signals, or smoothing."""
	def __init__(self, **kwargs):
		self.delay_amp    = kwargs["delay_amp"]
		self.delay_period = kwargs["delay_period"]
		self.delay_phase  = kwargs["delay_phase"]*utils.degree
		self.spin_period  = kwargs["spin_period"]
		self.spin_phase   = kwargs["spin_phase"]*utils.degree
		self.opening_angle= kwargs["opening_angle"]*utils.degree
		self.scan_period  = kwargs["scan_period"]
		self.scan_phase   = kwargs["scan_phase"]*utils.degree
		self.orbit_period = kwargs["orbit_period"]
		self.orbit_phase  = kwargs["orbit_phase"]*utils.degree
		self.orbit_step   = kwargs["orbit_step"]
		self.eclip_ang    = kwargs["eclib_ang"]*utils.degree
		self.ref_ctime    = kwargs["ref_ctime"]
	def calc_elements(self, ctime):
		"""Generate orbital elements for each ctime."""
		t = ctime - self.ref_ctime
		ang_orbit = self.orbit_phase   + 2*np.pi*np.floor(t/self.orbit_step)*self.orbit_step/self.orbit_period
		ang_scan  = self.scan_phase    + 2*np.pi*t/self.scan_period
		ang_spin  = self.spin_phase    + 2*np.pi*t/self.spin_period
		ang_delay = self.delay_phase   + 2*np.pi*t/self.delay_period
		return bunch.Bunch(ctime=ctime, orbit=ang_orbit, scan=ang_scan, spin=ang_spin, delay=ang_delay)
	def calc_orientation(self, elements):
		"""Compute a rotation matrix representing the orientation of the
		telescope for the given orbital elements."""
		R = np.eye(3)
		R = rot(R, "z", elements.spin)
		R = rot(R, "y", np.pi/2 - self.opening_angle)
		R = rot(R, "z", elements.scan)
		R = rot(R, "y", np.pi/2 - self.eclip_ang)
		R = rot(R, "z", elements.orbit)
		return R
	def calc_pointing(self, orientation, delay_ang, offset):
		"""Compute the on-sky pointing and phase delay angle for the given telescope
		orientation and phase delay, after applying the given offset from the
		boresight."""
		# Set up the offset
		Rb = np.eye(3)
		Rb = rot(Rb, "z", offset[0])
		Rb = rot(Rb, "y", offset[1])
		Rb = rot(Rb, "z", offset[2])
		# Form the total rotation matrix
		R = np.einsum("tij,jk->tik", orientation, Rb)
		# Switch to t-last ordering, as it is easier to work with
		R = np.einsum("tik->kit", R)
		# Extract the pointing angles
		xvec, zvec = R[:,0], R[:,2]
		point = utils.rect2ang(zvec,axis=0, zenith=False)
		# Make sure phi is between -180 and 180
		point[0] = utils.rewind(point[0])
		# Get the polarization orientation on the sky
		gamma = np.arctan2(xvec[2], -zvec[1]*xvec[0]+zvec[0]*xvec[1])
		# And the delay at each time
		delay = self.delay_amp * np.sin(delay_ang)
		return bunch.Bunch(point=point, gamma=gamma, delay=delay, pos=zvec)

##### Spectrogram generation #####

def calc_spectrogram(specmap, wcs_freq):
	"""Transform specmap into a spectrogram. Specmap must be be equi-spaced
	according to wcs_frea. Returns a spectrogram and corresponding delay_wcs."""
	return pixutils.spec2delay(spec, wcs_freq)

def calc_specmap(field, freqs, beam):
	"""Compute the observed spectrum for the given field, taking
	into account the observed beam and correcting for the beam present
	in the field data."""
	specmap = field(freqs)
	specmap = update_beam(specmap, freqs, beam, field.beam)
	return specmap

def update_beam(specmap, freqs, obeam, ibeam, apod=0.5):
	"""Return specmap after unapplying ibeam and applying obeam."""
	# Apodize a bit to reduce ringing
	apod_pix = int(0.5*max(ibeam.fwhm, obeam.fwhm)/(specmap.cdelt[0]*utils.degree))
	map  = specmap.apod(apod_pix, fill="mean")
	# Update the beams
	fmap = enmap.fft(map)
	lmap = enmap.lmap(map)
	fmap *= obeam(freqs, lmap)
	fmap /= ibeam(freqs, lmap)
	map = enmap.ifft(fmap).real
	return map

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
		self.map  = map
		self.pmap = utils.interpol_prefilter(map, order=order)
	def project(self, shape, wcs):
		"""Project our values onto a patch with the given shape and wcs. Returns
		a new Field."""
		pos = enmap.posmap(shape, wcs)
		map = self.pmap.at(pos, order=self.order, prefilter=False)
		return Field(self.name, map, self.spec, self.beam, self.order)
	def __call__(self, freq):
		return enmap.samewcs(self.spec(freq, self.map), self.map)

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
	fwhm = 0
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


##### Helpers #####

def rmul(R, a): return np.einsum("...ij,...jk->...ik",a,R)
def rmat(ax, ang): return utils.rotmatrix(ang, ax)
def rot(a, ax, ang): return rmul(rmat(ax,ang),a)
