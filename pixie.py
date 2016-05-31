import numpy as np, h5py, pixutils, astropy.io.fits, copy
from enlib import enmap, bunch, utils, bench

##### The main simulator class #####

class PixieSim:
	def __init__(self, config):
		# Pointing
		self.pointgen      = PointingGenerator(**config.__dict__)
		self.sample_period = config.sample_period

		# Frequencies
		self.nfreq         = int(config.nfreq)
		self.fmax          = int(config.fmax)
		self.lmax          = int(config.lmax)
		self.nl            = int(config.nl)

		# Spectrogram patches
		self.chunk_size    = int(config.chunk_size)
		self.bounds_skip   = int(config.bounds_skip)
		self.bounds_niter  = int(config.bounds_niter)
		self.beam_nsigma   = config.beam_nsigma
		self.patch_res     = config.patch_res * utils.degree

		# Subsampling. Force to odd number
		self.subsample_num = int(config.subsample_num)/2*2+1
		self.subsample_method = config.subsample_method

		# Beam types
		self.beam_types = [parse_beam(beam_type) for beam_type in config.beam_types]

		# Barrels
		self.barrels = [parse_barrel(barrel) for barrel in config.barrels]

		# Detectors
		self.dets = [parse_det(det) for det in config.dets]

		# Skies. Each sky is a list of fields, such as cmb or dust
		self.skies = [[read_field(field) for field in sky] for sky in config.skies]

		# Computed
		self.ncomp       = 3
		self.nhorn       = 2
		self.ndet        = len(self.dets)
		self.nsamp_orbit = int(np.round(self.pointgen.orbit_period/self.sample_period))
		self.wcs_freq    = pixutils.freq_geometry(self.fmax, self.nfreq)
		self.freqs       = self.wcs_freq.wcs_pix2world(np.arange(self.nfreq),0)[0]

		# We need the maps to be band-limited before we do anything else with
		# them. To do this we will pre-smooth everything to the smallest (highest
		# response) common beam, and update the beams to compensate. For this operation we need
		# raster beams, so pre-rasterize.
		refbeam = calc_reference_beam(self.beam_types, self.fmax, self.nfreq, self.lmax, self.nl)
		self.skies = [[field.to_beam(refbeam) for field in sky] for sky in self.skies]
	@property
	def ref_ctime(self): return self.pointgen.ref_ctime
	def gen_tod_orbit(self, i):
		"""Generate the full TOD for orbit #i, defined as the samples
		i*nsamp_orbit:(i+1)*nsamp_orbit. If the reference time is not
		at the start of an orbit, the cut from one orbit to the next
		will happen in the middle of these tods. We assume that there
		is an integer number of samples in an orbit."""
		samples = np.arange(i*self.nsamp_orbit,(i+1)*self.nsamp_orbit)
		return self.gen_tod_samples(samples)
	def gen_tod_samples(self, samples):
		"""Generate time ordered data for the given sample indices.
		The sample index starts at 0 at the reference time, and
		increases by 1 for each sample_period seconds. This function
		applies subchunking (to save memory) and subsampling (to simulate
		continous integration inside each sample) automatically."""
		tods   = []
		nsamp  = len(samples)
		ctimes = self.ref_ctime + np.asarray(samples)*float(self.sample_period)
		for i in range(0, nsamp, self.chunk_size):
			mytimes = ctimes[i:i+self.chunk_size]
			subtimes, weights = subsample(mytimes, self.subsample_num, self.subsample_method)
			subtod  = self.gen_tod_raw(subtimes)
			mytod   = downsample_tod_blockwise(subtod, weights)
			tods.append(mytod)
		return concatenate_tod(tods)
	def gen_tod_raw(self, ctimes):
		"""Generate the time-ordered data for the sample times given in ctimes.
		No oversampling is done, nor is chunking. This is a helper function that
		should usually not be called directly."""
		elements    = self.pointgen.calc_elements(ctimes)
		orientation = self.pointgen.calc_orientation(elements)
		# Prepare the input spectrogram patches
		patches = [[None for btype in self.beam_types] for sky in self.skies]
		for isky, sky in enumerate(self.skies):
			with bench.show("get_patch_bounds"):
				shape, wcs = self.get_patch_bounds(orientation)
			for ifield, field in enumerate(sky):
				with bench.show("field.project"):
					subfield = field.project(shape, wcs)
				for ibtype, btype in enumerate(self.beam_types):
					with bench.show("calc_specmap"):
						specmap = calc_specmap(subfield, self.freqs, btype)
					with bench.show("calc_spectrogram"):
						pmap, wcs_delay = calc_spectrogram(specmap, self.wcs_freq)
					# This should be elsewhere
					with bench.show("prefilter"):
						pmap = enmap.samewcs(np.ascontiguousarray(np.rollaxis(pmap,1)),pmap)
						pmap = utils.interpol_prefilter(pmap, npre=1, order=3)
					add_patch(patches, isky, ibtype, Patch(pmap, wcs_delay))
		# Add up the signal contributions to each horn
		horn_sig  = np.zeros([self.nhorn,self.ncomp,len(ctimes)])
		point_all, pix_all = [], []
		for ibarrel, barrel in enumerate(self.barrels):
			for isub, subbeam in enumerate(barrel.subbeams):
				patch = patches[barrel.sky][subbeam.type]
				with bench.show("calc_pointing"):
					point = self.pointgen.calc_pointing(orientation, elements.delay, subbeam.offset)
				with bench.show("calc_pixels"):
					pix   = calc_pixels(point.angpos, point.delay, patch.wcs, patch.wcs_delay)
				with bench.show("calc_response"):
					resp  = calc_response(point.gamma, subbeam.response, ibarrel) # [nhorn,{dc,delay},...]
				with bench.show("calc_horn_signal"):
					for ihorn in range(self.nhorn):
						horn_sig[ihorn] += calc_horn_signal(patch.map, pix, resp[ihorn])
				# Save pointing and pixels for TOD output
				point_all.append(point)
				pix_all.append(pix)
		# Read off each detector's response
		det_sig = np.zeros([self.ndet,len(ctimes)])
		with bench.show("calc_det_signal"):
			for idet, det in enumerate(self.dets):
				det_sig[idet] = calc_det_signal(horn_sig[det.horn], det.response)
		# output result as a PixieTOD
		return PixieTOD(
				signal   = det_sig,
				elements = [elements.ctime, elements.orbit, elements.scan, elements.spin, elements.delay],
				point    = [[point.angpos[0],point.angpos[1],point.gamma] for point in point_all],
				pix      = [pix[1:] for pix in pix_all])
	def get_patch_bounds(self, orientation):
		"""Return the (shape,wcs) geometry needed for simulating the sky at the
		given orientation, including any beam offsets and a margin to allow for
		beam smoothing."""
		# Get the padding needed based on the beam size and maximum beam offset
		rmax  = [subbeam.offset[1] for barrel in self.barrels for subbeam in barrel.subbeams]
		bsize = max([beam_type.fwhm for beam_type in self.beam_types])*utils.fwhm
		pad   = rmax + bsize*self.beam_nsigma
		# Get a (hopefully) representative subset of the pointing as angular coordinates
		zvec  = orientation.T[:,2,::self.bounds_skip]
		angpos= utils.rect2ang(zvec, axis=0, zenith=False)
		# Get the bounds needed for a longitudinal strip to contain these angles
		box   = pixutils.longitude_band_bounds(angpos, niter=self.bounds_niter)
		# 2*pad because the argument to widen_box is the total increase in width, not
		# the radius.
		box   = utils.widen_box(box, 2*pad, relative=False)
		# Actually generate the geometry
		return  pixutils.longitude_geometry(box, res=self.patch_res, dims=(self.ncomp,))

##### TOD container #####

# What members should we have in the TOD? The minimum is signal for each detector
# plus orbital elements. This is what the mapmaker would take as input. But it
# can also be useful to include the on-sky pointing (and perhaps pixels). I will
# include all of it for now. Store as signal[ndet,nsamp], elements[{ctime,orbit,scan,spin,delay},nsamp],
# point[nbeam,{phi,theta,gamma},nsamp], pix[nbeam,{y,x,delay},nsamp]

class PixieTOD:
	"""Container for Pixie TODs."""
	def __init__(self, signal, elements, point=None, pix=None):
		self.signal   = np.array(signal)
		self.elements = np.array(elements)
		self.point    = np.array(point) if point is not None else None
		self.pix      = np.array(pix)   if pix   is not None else None

def downsample_tod_blockwise(tod, weights):
		"""Return a new PixieTOD where the samples have been downsampled according
		to weights[nsub], such that the result is nsub times shorter."""
		# This may break for quantities with angle cuts!
		def dmean(a):
			if a is not None:
				return np.sum(a.reshape(a.shape[:-1]+(-1,weights.size))*weights,-1)
		def dmid(a):
			if a is not None:
				return a[...,len(weights)/2::len(weights)]
		return PixieTOD(dmean(tod.signal), dmid(tod.elements), dmid(tod.point), dmid(tod.pix))

def concatenate_tod(tods):
	"""Concatenate a list of tods into a single one."""
	return PixieTOD(
			np.concatenate([tod.signal   for tod in tods],-1),
			np.concatenate([tod.elements for tod in tods],-1),
			np.concatenate([tod.point    for tod in tods],-1) if tods[0].point is not None else None,
			np.concatenate([tod.pix      for tod in tods],-1) if tods[0].pix   is not None else None)

def write_tod(fname, tod):
	with h5py.File(fname, "w") as hfile:
		for key in ["signal","elements","point","pix"]:
			try:
				hfile[key] = getattr(tod, key)
			except AttributeError: pass

def read_tod(fname):
	data = {}
	with h5py.File(fname, "r") as hfile:
		for key in ["signal","elements","point","pix"]:
			if key in hfile:
				data[key] = tod[key].value
	return PixieTOD(**data)

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
	sig_dc    = utils.interpol(pmap, pix[[0,2,3]], order=order, mode="constant", mask_nan=False, prefilter=False)
	sig_delay = utils.interpol(pmap, pix[1:],      order=order, mode="constant", mask_nan=False, prefilter=False)
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
	ntime, nhorn, ncomp = gamma.size, 2, 3
	A, B = bidx, 1-bidx
	u = np.full(ntime, 1.0)
	o = np.zeros(ntime)
	c = np.cos(2*gamma)
	s = np.sin(2*gamma)
	# First apply the sky rotation
	R = np.array([
		[ u, o, o],
		[ o, c,-s],
		[ o, s, c]])
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
		self.eclip_angle  = kwargs["eclip_angle"]*utils.degree
		self.ref_ctime    = kwargs["ref_ctime"]
	def calc_elements(self, ctime):
		"""Generate orbital elements for each ctime."""
		t = ctime - self.ref_ctime
		ang_orbit = self.orbit_phase   + 2*np.pi*np.floor(t/self.orbit_step)*self.orbit_step/self.orbit_period
		ang_scan  = self.scan_phase    + 2*np.pi*t/self.scan_period
		ang_spin  = self.spin_phase    + 2*np.pi*t/self.spin_period
		ang_delay = self.delay_phase   + 2*np.pi*t/self.delay_period
		delay     = self.delay_amp * np.sin(ang_delay)
		return bunch.Bunch(ctime=ctime, orbit=ang_orbit, scan=ang_scan, spin=ang_spin, ang_delay=ang_delay, delay=delay)
	def calc_orientation(self, elements):
		"""Compute a rotation matrix representing the orientation of the
		telescope for the given orbital elements."""
		R = np.eye(3)
		R = rot(R, "z", elements.spin)
		R = rot(R, "y", np.pi/2 - self.opening_angle)
		R = rot(R, "z", elements.scan)
		R = rot(R, "y", np.pi/2 - self.eclip_angle)
		R = rot(R, "z", elements.orbit)
		return R
	def calc_pointing(self, orientation, delay, offset):
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

def update_beam(map, freqs, obeam, ibeam, apod=1.0):
	"""Return map after unapplying ibeam and applying obeam."""
	# Apodize a bit to reduce ringing, if requested
	if apod > 0:
		apod_pix = int(apod*max(ibeam.fwhm, obeam.fwhm)*utils.fwhm/(map.wcs.wcs.cdelt[0]*utils.degree))
		map  = map.apod(apod_pix, fill="mean")
	# Update the beams
	fmap = enmap.fft(map)
	lmap = np.sum(map.lmap()**2,0)**0.5
	fmap *= obeam(freqs, lmap)[:,None]
	fmap /= make_beam_safe(ibeam(freqs, lmap))[:,None]
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
		beam = Beam()
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
		map = enmap.ndmap(self.pmap.at(pos, order=self.order, prefilter=False, mask_nan=False),wcs)
		return Field(self.name, map, self.spec, self.beam, self.order)
	def to_beam(self, beam):
		"""Update our field to a new beam."""
		map = update_beam(self.map[None], [0], beam, self.beam, apod=0)[0]
		return Field(self.name, map, self.spec, beam, order=self.order)
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
		nonzero   = map[0] != 0
		polfrac   = map[1:]
		polfrac[:,nonzero] /= map[0,nonzero]
		res[:,1:] = res[:,0,None] * polfrac[None]
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

class Beam:
	type = "none"
	fwhm = 0
	def __call__(self, freq, l):
		freq, l = np.asarray(freq), np.asarray(l)
		return np.full(freq.shape + l.shape, 1.0)
	def copy(self): return copy.deepcopy(self)

class BeamGauss(Beam):
	type = "gauss"
	def __init__(self, fwhm):
		self.fwhm = fwhm
	def __call__(self, freq, l):
		freq, l = np.asarray(freq), np.asarray(l)
		res = np.zeros(freq.shape + l.shape)
		res[:] = np.exp(-0.5*l**2*(self.fwhm**2/(8*np.log(2))))
		return res

class BeamRaster(Beam):
	type = "raster"
	def __init__(self, vals, fmax, nfreq, lmax, nl):
		self.vals = vals
		self.fmax, self.nfreq = fmax, nfreq
		self.lmax, self.nl    = lmax, nl
		# Fit a gaussian by finding the l at which we are
		# down by 1/e**0.5
		mvals = np.mean(vals,0)
		idown = np.where(mvals<np.exp(-0.5))[0]
		idown = idown[0] if len(idown) > 0 else self.nl
		ldown = idown*float(lmax)/nl
		self.fwhm = (8*np.log(2))**0.5/ldown
	def __call__(self, freq, l):
		# Interpolate the beam value
		freq, l = np.asarray(freq), np.asarray(l)
		ifreq = freq*self.nfreq/self.fmax
		il    = l*self.nl/self.lmax
		i     = np.zeros((2,) + freq.shape + l.shape)
		i[0]  = ifreq[:,None]
		i[1]  = il[None,:]
		return utils.interpol(self.vals, i, order=3, mode="nearest")
	def __mul__(self, other):
		res = self.copy()
		res.vals *= other.vals
		return res
	def __div__(self, other):
		res = self.copy()
		good = (res.vals !=0) & (other.vals != 0)
		res.vals[good] /= other.vals[good]
		return res

def rasterize_beam(beam, fmax, nfreq, lmax, nl):
	"""Convert any beam into a rasterized version of itself, within
	the frequency range [0:fmax] and multipole range [0:lmax], with
	nfreq and nl samples in between respectively."""
	freq = np.arange(nfreq,dtype=float)*fmax/nfreq
	l    = np.arange(nl,dtype=float)*lmax/nl
	bval = beam(freq, l)
	return BeamRaster(bval, fmax, nfreq, lmax, nl)

def smallest_beam_raster(beams):
	"""Given a list of beams, all of which must be rasterized beams with the same raster,
	returns a single, frequency-independent raster beam which has higher resolution than
	all of them."""
	res  = beams[0].copy()
	res.vals = np.max([b.vals for b in beams],0)
	return res

def smallest_beam_freq(beam):
	"""Given a raster beam. Returns a new raster beam that is frequency-independent, and
	which has the highest resolution of all the frequencies of the input beam."""
	res = beam.copy()
	res.vals = np.max(res.vals,0)[None]
	res.nfreq = 1
	return res

def calc_reference_beam(beams, fmax, nfreq, lmax, nl):
	"""Given a list of beams of any type, compute the frequency-independent
	reference beam on the given raster in freq and multipole."""
	beams = [rasterize_beam(beam, fmax, nfreq, lmax, nl) for beam in beams]
	refbeam = smallest_beam_raster(beams)
	refbeam = smallest_beam_freq(refbeam)
	return refbeam

##### Containers #####

class Patch:
	"""A simple container for spectrogram patches. Should really have a
	more fullblown, general class for this ala enmap. But for now, this
	is all I need."""
	def __init__(self, map, wcs_delay):
		self.map, self.wcs_delay = map.copy(), wcs_delay.deepcopy()
	@property
	def wcs(self): return self.map.wcs
	def copy(self): return Patch(self.map, selv.wcs_delay)

##### Oversampling #####

def subsample(ctime, nsub, scheme="plain"):
	"""Given an *equispaced* ctime, generate a new ctime
	with nsub times more samples. The samples are placed
	depending on the scheme chosen, each of which corresponds
	to a different integration quadrature. Returns the new
	subsampled ctime and an array of quadrature weights."""
	nbin = len(ctime)
	dt   = ctime[1]-ctime[0]
	# Ignore schemes if subsampling is disabled (nsub=1)
	if nsub == 1: return ctime, np.full([nsub],1.0)
	# Get the subsample offsets and subsample weights
	# depending on which scheme we use.
	if scheme == "plain":
		off     = 0.5*((2*np.arange(nsub)+1)/float(nsub)-1)
		weights = np.full(nsub,1.0)/float(nsub)
	elif scheme == "trap":
		off = np.arange(nsub)/float(nsub-1)-0.5
		weights = np.concatenate([[1],np.full(nsub-2,2.0),[1]])/(2.0*(nsub-1))
	elif scheme == "simpson":
		# Only odd number of subsamples supported
		nsub = nsub/2*2+1
		off = np.arange(nsub)/float(nsub-1)-0.5
		weights = np.concatenate([[1],((1+np.arange(nsub-2))%2+1)*2,[1]])/(3.0*(nsub-1))
	elif scheme == "gauss":
		off, weights = np.polynomial.legendre.leggauss(nsub)
		# Go from [-1,1] to [-0.5,0.5]
		off /= 2
		weights /= 2
	# Build the subsampled version of ctime
	ctime_out = (ctime[:,None] + (off*dt)[None,:]).reshape(-1)
	return ctime_out, weights

##### Parsing #####

def parse_beam(params):
	if params["type"] is "none":
		return Beam()
	elif params["type"] is "gauss":
		return BeamGauss(params["fwhm"]*utils.degree)
	else:
		raise ValueError("Unknown beam type '%s'" % params["type"])

def parse_barrel(params):
	return bunch.Bunch(
			sky=params["sky"],
			subbeams=[bunch.Bunch(
					type = int(subbeam["type"]),
					offset = np.array(subbeam["offset"]),
					response = np.array(subbeam["response"]),
				) for subbeam in params["subbeams"]])

def parse_det(params):
	return bunch.Bunch(horn=int(params["horn"]), response=np.array(params["response"]))

##### Helpers #####

def rmul(R, a): return np.einsum("...ij,...jk->...ik",a,R)
def rmat(ax, ang): return utils.rotmatrix(ang, ax)
def rot(a, ax, ang): return rmul(rmat(ax,ang),a)
def add_patch(patches, isky, ibtype, patch):
	if patches[isky][ibtype] is None:
		patches[isky][ibtype] = patch
	else:
		patches[isky][ibtype].map += patch.map

