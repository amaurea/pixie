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
#          patch = patches[barrel.sky][subbeam.type]
#          point = calc_pointing(orbit, subbeam)
#          pix   = calc_pixels(point.angpos, point.delay, patch.wcs, patch.wcs_delay)
#          resp  = calc_response(point.gamma, subbeam.response, barrel) # [nhorn,{dc,delay},...]
#          for each horn:
#             horn_sig[horn] += calc_barrel_signal(patch, pix, resp[horn])
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
#
# It would have been convenient to have an enmap-like class supporting
# spectra wcs, so I don't have to pass around map,wcs tuples all the time.
# I should probably make a more general class for that. Most of enmap's
# features could be supported, except for the multipole-specific stuff,
# which requires knowledge about what the wcs axes mean.


##### The main simulator class #####

class PixieSim:
	def __init__(self, config):
		pass
	def gen_tod_raw(self, ctimes):
		"""Generate the time-ordered data for the sample times given in ctimes.
		No oversampling is done, nor is chunking. This is a helper function that
		should usually not be called directly."""
		elements    = self.pointgen.calc_elements(ctimes)
		orientation = self.pointgen.calc_orientation(elements)
		# Prepare the input spectrogram patches
		patches = None
		for isky, sky in enumerate(self.skies):
			shape, wcs = self.get_patch_bounds(orientation)
			for ifield, field in enumerate(sky):
				subfield = field.project(shape, wcs)
				for ibtype, btype in self.beam_types:
					specmap = calc_specmap(subfield, self.freqs, btype)
					pmap, wcs_delay = calc_spectrogram(specmap, self.wcs_freq)
					patches = add_patch(patches, isky, ibtype, Patch(pmap, wcs_delay))
		# Add up the signal contributions to each horn
		horn_sig = np.zeros([self.nhorn,self.ncomp,len(ctimes)])
		for ibarrel, barrel in self.barrels:
			for isub, subbeam in barrel.subbeams:
				patch = patches[barrel.sky, subbeam.type]
				point = self.pointgen.calc_pointing(orientation, elements.ang_delay, subbeam.offset)
				pix   = calc_pixels(point.angpos, point.delay, patch.wcs, patch.wcs_delay)
				resp  = calc_response(point.gamma, subbeam.response, ibarrel) # [nhorn,{dc,delay},...]
				for ihorn in range(self.nhorn):
					horn_sig[ihorn] += calc_horn_signal(patch.map, pix, resp[ihorn])
		# Read off each detector's response
		det_sig = np.zeros([self.ndet,len(ctimes)])
		for idet, det in enumerate(self.dets):
			det_sig[idet] = calc_det_signal(horn_sig[det.horn], det.response)
	def get_patch_bounds(self, orientation):
		"""Return the (shape,wcs) geometry needed for simulating the sky at the
		given orientation, including any beam offsets and a margin to allow for
		beam smoothing."""
		# Get the padding needed based on the beam size and maximum beam offset
		rmax  = [subbeam.offset[1] for subbeam in self.subbeams]
		bsize = max([beam_type.fwhm for beam_type in self.beam_types])*utils.fwhm
		pad   = rmax + bsize*self.beam_nsigma
		# Get a (hopefully) representative subset of the pointing as angular coordinates
		zvec  = orientation.T[:,2,::self.bounds_skip]
		angpos= utils.rect2ang(zvec, axis=0, zenith=False)
		# Get the bounds needed for a longitudinal strip to contain these angles
		box   = pixutils.longitude_band_bounds(angpos, niter=self.bounds_iter)
		box   = utils.widen_box(box, pad, relative=False)
		# Actually generate the geometry
		return  pixutils.longitude_geometry(box, res=self.patch_res, dims=(self.ncomp,))

##### Signal #####

def calc_det_signal(sig_horn, det_response):
	"""Calculate the signal a detector with det_response[{TQU}] sees,
	based on the signal sig_horn[{TQU},ntime] incident on its horn."""
	return np.einsum("a,at->t", det_response, sig_horn)

def calc_horn_signal(pmap, pix, horn_resp, order=3):
	"""Calcualte the TQU signal incident on a horn with responsitivy
	horn_resp[{dc,delay},{TQU},{TQU},ntime]. pmap is the patch
	array pix was computed for, and is [{dc,delay,yx},ntime]"""
	# Get the barrel-indicent signal for each sample
	sig_dc    = utils.interpol(pmap, pix[[0,2,3]], order=order, mode="constant", mask_nan=False)
	sig_delay = utils.interpol(pmap, pix[1:],      order=order, mode="constant", mask_nan=False)
	# Apply the horn's response matrix to get the signal entering that horn
	sig_horn  = np.einsum("abt,bt->at", horn_resp[0], sig_dc)
	sig_horn += np.einsum("abt,bt->at", horn_resp[1], sig_delay)
	return sig_horn

def calc_response(gamma, beam_resp, bidx):
	"""Calculate the response matrix transforming TQU on the sky
	seen through barrel #bidx, to TQU entering each horn,
	after taking coordinate transformations, interferometric mixing
	and beam responsitivity into account. beam_resp is [{TQU},{TQU}].
	"""
	ntime, nhorn, ncomp = gamma.shape, 2, 3
	A, B = bidx, 1-bidx
	u = np.full(ntime, 1.0)
	c = np.cos(2*gamma)
	s = np.sin(2*gamma)
	# First apply the sky rotation
	R = np.array([
		[ u, 0, 0],
		[ 0, c,-s],
		[ 0, s, c]])
	# Then apply the beam response
	R = np.einsum("ab,bct->act", beam_resp, R)
	# And then the interferometry:
	# I = 1/4*(Ii + Ij)[0] + 1/4*(Ii - Ij)[delta]
	# Q = 1/4*(Qi - Qj)[0] + 1/4*(Qi + Qj)[delta]
	# U = 1/4*(Ui - Uj)[0] + 1/4*(Ui + Uj)[delta]
	res = np.zeros([nhorn, 2, ncomp, ncomp, ntime])
	T, Q, U  = R
	res[A,0] = [ T, Q, U] # me, DC
	res[A,1] = [ T, Q, U] # me, delay
	res[B,0] = [ T,-Q,-U] # other, DC
	res[B,1] = [-T, Q, U] # other, delay
	res /= 4
	return res

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
		return bunch.Bunch(ctime=ctime, orbit=ang_orbit, scan=ang_scan, spin=ang_spin, ang_delay=ang_delay)
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
	def calc_pointing(self, orientation, ang_delay, offset):
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
		angpos = utils.rect2ang(zvec,axis=0, zenith=False)
		# Make sure phi is between -180 and 180
		angpos[0] = utils.rewind(angpos[0])
		# Get the polarization orientation on the sky
		gamma = np.arctan2(xvec[2], -zvec[1]*xvec[0]+zvec[0]*xvec[1])
		# And the delay at each time
		delay = self.delay_amp * np.sin(ang_delay)
		return bunch.Bunch(angpos=angpos, gamma=gamma, delay=delay, pos=zvec)

def calc_pixels(angpos, delay, wcs_pos, wcs_delay):
	"""Maps pointing to pixels[{pix_dc, pix_delay, pix_y, pix_x},ntime]."""
		ntime = angpos.shape[-1]
		pix = np.zeros([4,ntime])
		pix[0]  = wcs_delay.wcs_world2pix([0], 1)[0]
		pix[1]  = wcs_delay.wcs_world2pix(np.abs(delay), 1)[0]
		# angpos is [{phi,theta},ntime], but world2pix wants [:,{phi,theta}] in degrees
		pdeg = angpos.T / utils.degree
		# The result will be [:,{x,y}], but we want [{y,x},ntime]
		pix[3:1:-1] = wcs_pos.wcs_world2pix(pdeg,0).T
		return pix

##### Spectrogram generation #####

def calc_spectrogram(specmap, wcs_freq):
	"""Transform specmap into a spectrogram. Specmap must be be equi-spaced
	according to wcs_frea. Returns a spectrogram and corresponding delay_wcs."""
	return pixutils.spec2delay(specmap, wcs_freq)

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
	fmap /= make_beam_safe(ibeam(freqs, lmap))
	map = enmap.ifft(fmap).real
	return map

def make_beam_safe(beam, tol=1e-13):
	"""Cap very low beam values from below while preserving sign.
	Not sure if preserving sign is important. The main goal of this
	function is to avoid division by zero or tiny numbers when unapplying
	a beam."""
	beam = np.array(beam)
	bad  = np.abs(beam)<tol
	sig  = np.sign(beam[bad])
	sig[sig==0] = 1
	beam[bad] = tol*sig
	return beam

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
		map = self.pmap.at(pos, order=self.order, prefilter=False, mask_nan=False)
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
