import numpy as np, h5py, astropy.io.fits, copy, enlib.wcs, warnings, logging, imp, scipy.signal
from enlib import enmap, bunch, utils, bench, powspec, fft, sharp, coordinates, curvedsky, lensing, aberration
L = logging.getLogger(__name__)

# We currently use a normal fits WCS to set up the local patches along
# each chunk of the scan. A problem with this is that the coordinates
# still change rapidly around the pole even if the pixels don't. That
# leads to Q and U looking like quadrupoles around the pole, and hence
# makes it difficult to apply beam smoothing.
#
# Could either
# 1. Introduce a special case for polarization angles, rotating
#    them to be relative to the local y axis.
# 2. Map the whole chunk system to the equator via an extra
#    coordinate rotation.
# Both of these require an extra piece of information to be passed
# with the shape,wcs that defines the patch.

##### The main simulator classes #####

# Simulation steps:
# 1. Optics and detector inputs (current PixieSim)
# 2. Subsequent processing before readout
# Step #2 includes the tod filter and noise generation,
# and any glitches and frequency spikes, etc.
# This would make sense as a separate class, which doesn't
# need to care about all the difficulties with the optics.
#
# Sensible names for these?
# 1. OpticsSim
# 2. ReadoutSim

class PixieSim:
	"""Top-level class that implements the whole TOD simulation process."""
	def __init__(self, config):
		self.optics_sim  = OpticsSim(config)
		self.readout_sim = ReadoutSim(config)
	def sim_tod(self, i):
		tod = self.optics_sim.gen_tod_orbit(i)
		tod = self.readout_sim.sim_readout(tod)
		return tod

class OpticsSim:
	def __init__(self, config):
		"""Initialize a Pixie Optics simulator from the given configuration object.
		This version of OpticsSim does not support frequency-dependent beams. This
		should lead to a large simplification and speedup."""
		# Pointing
		self.pointgen      = PointingGenerator(**config.__dict__)
		self.sample_period = config.sample_period

		# Frequencies
		self.nfreq         = int(config.nfreq)
		self.fmax          = int(config.fmax)
		self.lmax          = int(config.lmax)
		self.nl            = int(config.nl)

		# Subsampling. Force to odd number
		self.subsample_num = int(config.subsample_num)/2*2+1
		self.subsample_method = config.subsample_method
		self.chunk_size    = int(config.chunk_size)

		# Beam types
		self.beams = [parse_beam(beam) for beam in config.beams]

		# Filter types
		self.filters = [parse_filter(filter) for filter in config.filters]

		# Barrels
		self.barrels = [parse_barrel(barrel) for barrel in config.barrels]

		# Detectors
		self.dets = [parse_det(det) for det in config.dets]

		# Skies. Each sky is a list of fields, such as cmb or dust
		self.skies = [[read_field(field) for field in sky] for sky in config.skies]

		# Precompute effective sky for each sky-beam-filter combination.
		with bench.show("setup_subskies"):
			self.subskies = self.setup_subskies(self.skies, self.barrels, self.beams, self.filters)

		# Computed
		self.ncomp       = 3
		self.nhorn       = 2
		self.ndet        = len(self.dets)
		self.nsamp_orbit = int(np.round(self.pointgen.scan_period/self.sample_period))
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
			L.debug("chunk %3d/%d" % (i/self.chunk_size, (nsamp+self.chunk_size-1)/self.chunk_size))
			# +1 for the extra overlap-sample, which we will use to remove
			# any discontinuities caused by interpolation issues.
			mytimes = ctimes[i:i+self.chunk_size+1]
			subtimes, weights = subsample(mytimes, self.subsample_num, self.subsample_method)
			subtod  = self.gen_tod_raw(subtimes)
			with bench.show("downsample"):
				mytod   = downsample_tod_blockwise(subtod, weights)
			del subtod, subtimes, weights, mytimes
			tods.append(mytod)
		return concatenate_tod(tods)
	def gen_tod_raw(self, ctimes):
		"""Generate the time-ordered data for the sample times given in ctimes.
		No oversampling is done, nor is chunking. This is a helper function that
		should usually not be called directly. The result has units W/sr/m^2."""
		with bench.show("calc_elements"):
			elements    = self.pointgen.calc_elements(ctimes)
		with bench.show("calc_orientation"):
			orientation = self.pointgen.calc_orientation(elements)
		# Add up the signal contributions to each horn
		horn_sig  = np.zeros([self.nhorn,self.ncomp,len(ctimes)])
		for ibarrel, barrel in enumerate(self.barrels):
			for isub, sub in enumerate(barrel):
				sky = self.subskies[(sub.sky,sub.beam,sub.filter)]
				with bench.show("calc_pointing"):
					point = self.pointgen.calc_pointing(orientation, elements.delay, sub.offset)
				with bench.show("calc_sky_signal"):
					sky_sig = calc_sky_signal(sky, point.angpos, point.delay)
					del point.angpos, point.delay
				with bench.show("calc_response"):
					resp  = calc_response(point.gamma, sub.response, ibarrel) # [nhorn,{dc,delay},...]
					del point
				with bench.show("calc_horn_signal"):
					for ihorn in range(self.nhorn):
						horn_sig[ihorn] += calc_horn_signal(sky_sig, resp[ihorn])
					del resp, sky_sig
		del orientation
		# Read off each detector's response
		det_sig = np.zeros([self.ndet,len(ctimes)])
		with bench.show("calc_det_signal"):
			for idet, det in enumerate(self.dets):
				det_sig[idet] = calc_det_signal(horn_sig[det.horn], det.response)
		del horn_sig
		# output result as a PixieTOD
		return PixieTOD(
				signal   = det_sig,
				elements = [elements.ctime, elements.orbit, elements.scan, elements.spin, elements.delay])
	def setup_subskies(self, skies, barrels, beams, filters):
		"""Helper function for setting up the effective sky each
		sub-beam in each barrel sees."""
		# We cache the beam-smoothed skies, since beam-smoothing is slow
		# while applying a new filter is fast. If to_filter is set up
		# so that the map itself isn't copied when a new filter is
		# derived, this setup will also avoid wasting memory storing
		# skies that only differ in filter twice.
		res, beam_smoothed = {}, {}
		for barrel in barrels:
			for sub in barrel:
				btag = (sub.sky,sub.beam)
				if btag not in beam_smoothed:
					beam = beams[sub.beam]
					beam_smoothed[btag] = [field.to_beam(beam) for field in skies[sub.sky]]
				ftag = (sub.sky,sub.beam,sub.filter)
				if ftag not in res:
					filter = filters[sub.filter]
					res[ftag] = [field.to_filter(filter) for field in beam_smoothed[btag]]
		return res

#class OpticsSimOld:
#	def __init__(self, config):
#		"""Initialize a Pixie Optics simulator from the given configuration object."""
#		# Pointing
#		self.pointgen      = PointingGenerator(**config.__dict__)
#		self.sample_period = config.sample_period
#
#		# Frequencies
#		self.nfreq         = int(config.nfreq)
#		self.fmax          = int(config.fmax)
#		self.lmax          = int(config.lmax)
#		self.nl            = int(config.nl)
#
#		# Spectrogram patches
#		self.chunk_size    = int(config.chunk_size)
#		self.bounds_skip   = int(config.bounds_skip)
#		self.bounds_niter  = int(config.bounds_niter)
#		self.beam_nsigma   = config.beam_nsigma
#		self.patch_res     = config.patch_res * utils.degree
#		self.patch_nsub    = int(config.patch_nsub)
#		self.patch_pad     = config.patch_pad * utils.degree
#		self.patch_apod    = config.patch_apod * utils.degree
#
#		# Subsampling. Force to odd number
#		self.subsample_num = int(config.subsample_num)/2*2+1
#		self.subsample_method = config.subsample_method
#
#		# Beam types
#		self.beam_types = [parse_beam(beam_type) for beam_type in config.beam_types]
#
#		# Barrels
#		self.barrels = [parse_barrel(barrel) for barrel in config.barrels]
#
#		# Detectors
#		self.dets = [parse_det(det) for det in config.dets]
#
#		# Skies. Each sky is a list of fields, such as cmb or dust
#		self.skies = [[polrot_field(read_field(field)) for field in sky] for sky in config.skies]
#
#		# Computed
#		self.ncomp       = 3
#		self.nhorn       = 2
#		self.ndet        = len(self.dets)
#		self.nsamp_orbit = int(np.round(self.pointgen.scan_period/self.sample_period))
#		self.wcs_freq    = freq_geometry(self.fmax, self.nfreq)
#		self.freqs       = self.wcs_freq.wcs_pix2world(np.arange(self.nfreq),0)[0]
#		self.refbeam     = calc_reference_beam(self.beam_types, self.fmax, self.nfreq, self.lmax, self.nl)
#	@property
#	def ref_ctime(self): return self.pointgen.ref_ctime
#	def gen_tod_orbit(self, i):
#		"""Generate the full TOD for orbit #i, defined as the samples
#		i*nsamp_orbit:(i+1)*nsamp_orbit. If the reference time is not
#		at the start of an orbit, the cut from one orbit to the next
#		will happen in the middle of these tods. We assume that there
#		is an integer number of samples in an orbit."""
#		samples = np.arange(i*self.nsamp_orbit,(i+1)*self.nsamp_orbit)
#		return self.gen_tod_samples(samples)
#	def gen_tod_samples(self, samples):
#		"""Generate time ordered data for the given sample indices.
#		The sample index starts at 0 at the reference time, and
#		increases by 1 for each sample_period seconds. This function
#		applies subchunking (to save memory) and subsampling (to simulate
#		continous integration inside each sample) automatically."""
#		tods   = []
#		nsamp  = len(samples)
#		ctimes = self.ref_ctime + np.asarray(samples)*float(self.sample_period)
#		for i in range(0, nsamp, self.chunk_size):
#			L.debug("chunk %3d/%d" % (i/self.chunk_size, (nsamp+self.chunk_size-1)/self.chunk_size))
#			# +1 for the extra overlap-sample, which we will use to remove
#			# any discontinuities caused by interpolation issues.
#			mytimes = ctimes[i:i+self.chunk_size+1]
#			subtimes, weights = subsample(mytimes, self.subsample_num, self.subsample_method)
#			subtod  = self.gen_tod_raw(subtimes)
#			with bench.show("downsample"):
#				mytod   = downsample_tod_blockwise(subtod, weights)
#			del subtod, subtimes, weights, mytimes
#			tods.append(mytod)
#		return concatenate_tod(tods)
#	def gen_tod_raw(self, ctimes):
#		"""Generate the time-ordered data for the sample times given in ctimes.
#		No oversampling is done, nor is chunking. This is a helper function that
#		should usually not be called directly. The result has units W/sr/m^2."""
#		elements    = self.pointgen.calc_elements(ctimes)
#		orientation = self.pointgen.calc_orientation(elements)
#		patches     = self.calc_patch_signal(orientation) # W/sr/m^2
#		# Add up the signal contributions to each horn
#		horn_sig  = np.zeros([self.nhorn,self.ncomp,len(ctimes)])
#		for ibarrel, barrel in enumerate(self.barrels):
#			for isub, subbeam in enumerate(barrel.subbeams):
#				patch = patches[barrel.sky][subbeam.type]
#				with bench.show("calc_pointing"):
#					point = self.pointgen.calc_pointing(orientation, elements.delay, subbeam.offset)
#				with bench.show("calc_pixels"):
#					pix   = calc_pixels(point.angpos, point.delay, patch.wcs, patch.wcs_delay)
#					del point.angpos, point.delay
#				with bench.show("calc_response"):
#					resp  = calc_response(point.gamma, subbeam.response, ibarrel) # [nhorn,{dc,delay},...]
#					del point
#				with bench.show("calc_horn_signal"):
#					for ihorn in range(self.nhorn):
#						horn_sig[ihorn] += calc_horn_signal(patch.map, pix, resp[ihorn])
#					del pix, resp
#		del patches, orientation
#		# Read off each detector's response
#		det_sig = np.zeros([self.ndet,len(ctimes)])
#		with bench.show("calc_det_signal"):
#			for idet, det in enumerate(self.dets):
#				det_sig[idet] = calc_det_signal(horn_sig[det.horn], det.response)
#		del horn_sig
#		# output result as a PixieTOD
#		return PixieTOD(
#				signal   = det_sig,
#				elements = [elements.ctime, elements.orbit, elements.scan, elements.spin, elements.delay])
#	def get_patch_bounds(self, orientation):
#		"""Return the (shape,wcs) geometry needed for simulating the sky at the
#		given orientation, including any beam offsets and a margin to allow for
#		beam smoothing."""
#		# Get the padding needed based on the beam size and maximum beam offset
#		rmax  = [subbeam.offset[1] for barrel in self.barrels for subbeam in barrel.subbeams]
#		# Max beam size of any output beam
#		bsize = max([beam_type.fwhm for beam_type in self.beam_types])*utils.fwhm
#		# Reduce by amount already smoothed. Since fwhms are just approximate,
#		# cap to 1/4 of the original beam for the purposes of finding the padding area
#		bsize = (np.maximum((bsize/4)**2, bsize**2-self.refbeam.fwhm**2))**0.5
#		pad   = rmax + bsize*self.beam_nsigma + self.patch_apod
#		# Get a (hopefully) representative subset of the pointing as angular coordinates
#		zvec  = orientation.T[:,2,::self.bounds_skip]
#		angpos= utils.rect2ang(zvec, axis=0, zenith=False)
#		# Get the bounds needed for a longitudinal strip to contain these angles
#		box   = longitude_band_bounds(angpos, niter=self.bounds_niter)
#		# 2*pad because the argument to widen_box is the total increase in width, not
#		# the radius.
#		box   = utils.widen_box(box, 2*pad, relative=False)
#		# Actually generate the geometry
#		return  longitude_geometry(box, res=self.patch_res, dims=(self.ncomp,), ref=(0,0))
#	def calc_patch_signal(self, orientation):
#		"""Computes the per-patch interferograms in units W/sr/m^2."""
#		# Prepare the input spectrogram patches
#		patches = [[None for btype in self.beam_types] for sky in self.skies]
#		with bench.show("get_patch_bounds"):
#			shape, wcs = self.get_patch_bounds(orientation)
#		for isky, sky in enumerate(self.skies):
#			# Skip skies that aren't in use
#			if not any([barrel.sky == isky for barrel in self.barrels]): continue
#			for ifield, field in enumerate(sky):
#				with bench.show("field.project"):
#					subfield = calc_subfield(field, shape, wcs, self.refbeam,
#							subsample=self.patch_nsub, pad=self.patch_pad, apod=self.patch_apod)
#				for ibtype, btype in enumerate(self.beam_types):
#					with bench.show("calc_specmap"):
#						# W/sr/m^2/Hz
#						specmap = calc_specmap(subfield, self.freqs, btype, apod=self.patch_apod)
#					with bench.show("calc_spectrogram"):
#						# W/sr/m^2
#						pmap, wcs_delay = calc_spectrogram(specmap, self.wcs_freq)
#						del specmap
#					# This should be elsewhere
#					with bench.show("prefilter"):
#						pmap = enmap.samewcs(np.ascontiguousarray(np.rollaxis(pmap,1)),pmap)
#						pmap = utils.interpol_prefilter(pmap, npre=1, order=3)
#					add_patch(patches, isky, ibtype, Patch(pmap, wcs_delay))
#				del subfield, pmap
#		return patches

class ReadoutSim:
	"""This class handles the signal from it has hit the detectors and until
	it is read out. This is stuff like the tod highpass and lowpass, noise
	injection, glitches, etc."""
	def __init__(self, config):
		self.filters = [parse_tod_filter(f) for f in config.tod_filters]
	def sim_readout(self, tod):
		"""The main function, which applies all the readout effects, returning
		the TOD one would actually see."""
		# Only the frequency filter for now
		tod = self.apply_filters(tod)
		return tod
	def apply_filters(self, tod):
		res  = tod.copy()
		# Prepare our fourier space. All filters are assumed to be defined
		# in fourier space.
		dt   = (tod.elements[0,-1]-tod.elements[0,0])/(tod.nsamp-1)
		freqs= fft.rfftfreq(tod.nsamp, dt)
		fsig = fft.rfft(res.signal)
		# Actually apply the filters
		for f in self.filters:
			fsig *= f(freqs)
		fft.ifft(fsig, res.signal, normalize=True)
		return res

##### TOD container #####

# What members should we have in the TOD? The minimum is signal for each detector
# plus orbital elements. This is what the mapmaker would take as input. But it
# can also be useful to include the on-sky pointing (and perhaps pixels). I will
# include all of it for now. Store as signal[ndet,nsamp], elements[{ctime,orbit,scan,spin,delay},nsamp],
# point[nbeam,{phi,theta,gamma},nsamp], pix[nbeam,{y,x,delay},nsamp]

class PixieTOD:
	"""Container for Pixie TODs. Contains
	signal[ndet,nsamp]
	elements[{ctime,orbit,scan,spin,selay},nsamp]
	point[{phi,theta,gamma},nsamp] (optional)
	pix[{idelay,y,x},nsamp] (optional)."""
	def __init__(self, signal, elements, point=None, pix=None):
		self.signal   = np.array(signal)
		self.elements = np.array(elements)
		self.point    = np.array(point) if point is not None else None
		self.pix      = np.array(pix)   if pix   is not None else None
	def copy(self): return PixieTOD(self.signal, self.elements, point=self.point, pix=self.pix)
	@property
	def ndet(self): return self.signal.shape[0]
	@property
	def nsamp(self): return self.signal.shape[1]

def downsample_tod_blockwise(tod, weights, bsize=0x1000):
		"""Return a new PixieTOD where the samples have been downsampled according
		to weights[nsub], such that the result is nsub times shorter."""
		# This may break for quantities with angle cuts!
		def dmean(a):
			if a is not None:
				a   = a.reshape(a.shape[:-1] + (-1, weights.size))
				nsub= a.shape[-2]
				res = np.zeros(a.shape[:-1])
				for i1 in range(0, nsub, bsize):
					i2 = min(i1+bsize,nsub)
					res[...,i1:i2] = np.sum(a[...,i1:i2,:]*weights,-1)
				return res
				#return np.sum(a.reshape(a.shape[:-1]+(-1,weights.size))*weights,-1)
		def dmid(a):
			if a is not None:
				return a[...,len(weights)/2::len(weights)].copy()
		return PixieTOD(dmean(tod.signal), dmid(tod.elements), dmid(tod.point), dmid(tod.pix))

def concatenate_tod(tods):
	"""Concatenate a list of tods into a single one."""
	return PixieTOD(
			overlap_cat([tod.signal   for tod in tods],-1),
			overlap_cat([tod.elements for tod in tods],-1),
			overlap_cat([tod.point    for tod in tods],-1) if tods[0].point is not None else None,
			overlap_cat([tod.pix      for tod in tods],-1) if tods[0].pix   is not None else None)

def write_tod(fname, tod):
	with h5py.File(fname, "w") as hfile:
		for key in ["signal","elements","point","pix"]:
			try:
				data = getattr(tod, key)
				if data is None: continue
				hfile[key] = data
			except AttributeError: pass

def read_tod(fname, nsamp=None):
	data = {}
	with h5py.File(fname, "r") as hfile:
		for key in ["signal","elements","point","pix"]:
			if key in hfile:
				data[key] = hfile[key].value
				# Only read nsamp samples if requested.
				if nsamp is not None:
					data[key] = data[key][...,:nsamp]
	return PixieTOD(**data)

##### Signal #####

def calc_det_signal(sig_horn, det_response):
	"""Calculate the signal a detector with det_response[{TQU}] sees,
	based on the signal sig_horn[{TQU},ntime] incident on its horn."""
	return np.einsum("a,at->t", det_response, sig_horn)

def calc_horn_signal(sky_sig, horn_resp):
	"""Calcualte the TQU signal incident on a horn with responsitivy
	horn_resp[{dc,delay},{TQU},{TQU},ntime]. sky_sig is
	[{dc,delay},{TQU},ntime], and is computed by calc_sky_signal."""
	# Apply the horn's response matrix to get the signal entering that horn
	sig_horn = np.einsum("Dabt,Dbt->at", horn_resp, sky_sig)
	return sig_horn

def calc_sky_signal(sky, angpos, delay):
	"""Given a list of fields sky, interpolate each field's
	spectrogram at the given angpos[{phi,theta},ntime] and
	delay[ntime]. Returns sky_signal[{dc,delay},{TQU},ntime]."""
	ntime      = angpos.shape[-1]
	sky_signal = np.zeros([3,2,ntime])
	# Massage inputs for the field interpolation. We want to evaluate
	# both the DC and delayed tcorr at the same time, and angpos needs
	# to be in enmaps [theta,phi] ordering.
	delay  = np.array([delay*0,delay])
	angpos = angpos[::-1,None,:]
	for field in sky:
		sky_signal += field.at(delay, angpos, type="tcorr")
	# Move {dc,delay} axis first
	sky_signal = np.rollaxis(sky_signal, 1)
	return sky_signal

#def calc_pixels(angpos, delay, wcs_pos, wcs_delay):
#	"""Maps pointing to pixels[{pix_dc, pix_delay, pix_y, pix_x},ntime]."""
#	ntime = angpos.shape[-1]
#	pix = np.zeros([4,ntime])
#	pix[0]  = wcs_delay.wcs_world2pix([0], 0)[0]
#	pix[1]  = wcs_delay.wcs_world2pix(np.abs(delay), 0)[0]
#	# angpos is [{phi,theta},ntime], but world2pix wants [:,{phi,theta}] in degrees
#	pdeg = angpos.T / utils.degree
#	# The result will be [:,{x,y}], but we want [{y,x},ntime]
#	pix[3:1:-1] = wcs_pos.wcs_world2pix(pdeg,0).T
#	return pix

#def calc_horn_signal(pmap, pix, horn_resp, order=3):
#	"""Calcualte the TQU signal incident on a horn with responsitivy
#	horn_resp[{dc,delay},{TQU},{TQU},ntime]. pmap is the patch
#	array pix was computed for, and is [{dc,delay,yx},ntime]"""
#	# Get the barrel-indicent signal for each sample
#	sig_dc    = utils.interpol(pmap, pix[[0,2,3]], order=order, mode="constant", mask_nan=False, prefilter=False)
#	sig_delay = utils.interpol(pmap, pix[1:],      order=order, mode="constant", mask_nan=False, prefilter=False)
#	# Apply the horn's response matrix to get the signal entering that horn
#	sig_horn  = np.einsum("abt,bt->at", horn_resp[0], sig_dc)
#	sig_horn += np.einsum("abt,bt->at", horn_resp[1], sig_delay)
#	return sig_horn

def calc_response(gamma, beam_resp, bidx, bsize=0x10000):
	"""Calculate the response matrix transforming TQU on the sky
	seen through barrel #bidx, to TQU entering each horn,
	after taking coordinate transformations, interferometric mixing
	and beam responsitivity into account. beam_resp is [{TQU},{TQU}].
	"""
	ntime, nhorn, ncomp = gamma.size, 2, 3
	res = np.zeros([nhorn, 2, ncomp, ncomp, ntime])
	A, B = bidx, 1-bidx
	# Calc in chunks to save memory
	for i1 in range(0, ntime, bsize):
		i2   = min(i1+bsize,ntime)
		u = np.full(i2-i1, 1.0)
		o = np.zeros(i2-i1)
		c = np.cos(2*gamma[i1:i2])
		s = np.sin(2*gamma[i1:i2])
		# First apply the sky rotation
		R = np.array([
			[ u, o, o],
			[ o, c,-s],
			[ o, s, c]])
		del u,o,c,s
		# Then apply the beam response
		R = np.einsum("ab,bct->act", beam_resp, R)
		# And then the interferometry:
		# I = 1/4*(Ii + Ij)[0] + 1/4*(Ii - Ij)[delta]
		# Q = 1/4*(Qi - Qj)[0] + 1/4*(Qi + Qj)[delta]
		# U = 1/4*(Ui - Uj)[0] + 1/4*(Ui + Uj)[delta]
		T, Q, U  = R
		res[A,0,:,:,i1:i2] = [ T, Q, U] # me, DC
		res[A,1,:,:,i1:i2] = [ T, Q, U] # me, delay
		res[B,0,:,:,i1:i2] = [ T,-Q,-U] # other, DC
		res[B,1,:,:,i1:i2] = [-T, Q, U] # other, delay
	# This factor 4 is what we actually measure, but we
	# must remember to take it into account when making maps.
	# It's effectively part of the gain of the instrument.
	res /= 4
	return res

##### Pointing #####

class PointingGenerator:
	"""This class handles the instrument's pointing, but does not care
	about details like sampling, signals, or smoothing."""
	def __init__(self, **kwargs):
		self.delay_amp    = kwargs["delay_amp"]
		self.delay_period = kwargs["delay_period"]
		self.delay_shape  = kwargs["delay_shape"]
		self.delay_phase  = kwargs["delay_phase"]*utils.degree
		self.spin_period  = kwargs["spin_period"]
		self.spin_phase   = kwargs["spin_phase"]*utils.degree
		self.opening_angle= kwargs["opening_angle"]*utils.degree
		self.scan_period  = kwargs["scan_period"]
		self.scan_phase   = kwargs["scan_phase"]*utils.degree
		self.orbit_period = kwargs["orbit_period"]
		self.orbit_phase  = kwargs["orbit_phase"]*utils.degree
		self.orbit_step   = kwargs["orbit_step"]
		self.orbit_step_dur=kwargs["orbit_step_dur"]
		self.eclip_angle  = kwargs["eclip_angle"]*utils.degree
		self.ref_ctime    = kwargs["ref_ctime"]
	def calc_elements(self, ctime):
		"""Generate orbital elements for each ctime."""
		t     = ctime - self.ref_ctime
		## Get our orbit number, taking into account the transition time
		#orbit = np.floor((t+0.5*self.orbit_step_dur)/(self.orbit_step + self.orbit_step_dur))
		## Compensate for the transition time, so that the non-transition stuff
		## effectively gives a smooth orbit.
		#t    -= orbit * self.orbit_step_dur
		#ang_orbit = self.orbit_phase   + 2*np.pi*orbit*self.orbit_step/self.orbit_period
		ang_orbit = self.orbit_phase   + 2*np.pi*np.floor(t/self.orbit_step)*self.orbit_step/self.orbit_period
		ang_scan  = self.scan_phase    + 2*np.pi*t/self.scan_period
		ang_spin  = self.spin_phase    + 2*np.pi*t/self.spin_period
		ang_delay = self.delay_phase   + 2*np.pi*t/self.delay_period
		if self.delay_shape == "sin":
			delay = self.delay_amp * np.sin(ang_delay)
		elif self.delay_shape == "triangle":
			delay = self.delay_amp * scipy.signal.sawtooth(ang_delay+np.pi/2, 0.5)
		else: raise ValueError("Unrecognized delay shape '%s'" % self.delay_shape)
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
		# Apply our extra polarization rotation. See polrot_field.
		gamma -= angpos[0]*np.sin(angpos[1])
		#gamma -= angpos[0]
		return bunch.Bunch(angpos=angpos, gamma=gamma, delay=delay, pos=zvec)

def calc_pixels(angpos, delay, wcs_pos, wcs_delay):
	"""Maps pointing to pixels[{pix_dc, pix_delay, pix_y, pix_x},ntime]."""
	ntime = angpos.shape[-1]
	pix = np.zeros([4,ntime])
	pix[0]  = wcs_delay.wcs_world2pix([0], 0)[0]
	pix[1]  = wcs_delay.wcs_world2pix(np.abs(delay), 0)[0]
	# angpos is [{phi,theta},ntime], but world2pix wants [:,{phi,theta}] in degrees
	pdeg = angpos.T / utils.degree
	# The result will be [:,{x,y}], but we want [{y,x},ntime]
	pix[3:1:-1] = wcs_pos.wcs_world2pix(pdeg,0).T
	return pix

##### Spectrogram generation #####

def apply_beam_fullsky(map, new_beam, old_beam=None, lmax=None, ltest=10000, vtest=1e-10):
	"""Apply the beam new_beam to map using spherical
	harmonics."""
	# Prepare the map
	if map.wcs.wcs.cdelt[0] > 0: map = map[...,:,::-1]
	if map.wcs.wcs.cdelt[1] < 0: map = map[...,::-1,:]
	map = enmap.samewcs(np.ascontiguousarray(map), map)
	# Prepare b_l
	if old_beam is None: old_beam = lambda l: l*0+1
	# Find necessary lmax if none was specified
	if lmax is None:
		l    = np.arange(ltest+1.)
		beam = new_beam(l)/old_beam(l)
		mask = np.where(beam<vtest)[0]
		lmax = mask[0] if len(mask) > 0 else ltest
		beam = beam[:lmax+1]
	else:
		l    = np.arange(lmax+1.)
		beam = new_beam(l)/old_beam(l)
	# Prepare the SHT
	minfo = sharp.map_info_clenshaw_curtis(map.shape[-2], map.shape[-1])
	ainfo = sharp.alm_info(lmax=lmax)
	sht   = sharp.sht(minfo, ainfo)
	alm   = np.zeros((3,ainfo.nelem),dtype=complex)
	# Forwards transform
	sht.map2alm(map[:1].reshape(1,-1), alm[:1])
	sht.map2alm(map[1:].reshape(2,-1), alm[1:], spin=2)
	alm   = ainfo.lmul(alm, beam)
	# And transform back again
	sht.alm2map(alm[:1], map[:1].reshape(1,-1))
	sht.alm2map(alm[1:], map[1:].reshape(2,-1), spin=2)
	return map

def read_field(fname):
	"""Read a field from a specially prepared fits file."""
	map    = enmap.read_map(fname)
	header = astropy.io.fits.open(fname)[0].header
	name   = header["NAME"]
	# Get the spectral information and build a Spectrum object
	spec_type = header["SPEC"]
	if spec_type == "BLACK":
		# The reference temperature is the temperature we will
		# Taylor expand around, to it should be as representative as possible
		Tref = np.mean(map[0,::10,::10])
		spec = SpecBlackbody(name, Tref=Tref)
	elif spec_type == "GRAY":
		spec = SpecGraybody(name, T=float(header["TBODY"]), beta=float(header["BETA"]),
				fref=float(header["FREF"]), unit=float(header["SUNIT"]))
	else: raise ValueError("Spectrum type '%s' not recognized" % spec_type)
	# Get the beam information
	beam_type = header["BEAM"]
	if beam_type == "NONE":
		beam = Beam()
	elif beam_type == "GAUSS":
		beam = BeamGauss(float(header["FWHM"])*utils.degree*utils.fwhm)
	else: raise ValueError("Beam type '%s' not recognized" % beam_type)
	return Field(name.lower(), map, spec, beam)

def polrot_field(field):
	"""Apply a global polarization rotation to the given field to make
	it easier to smooth and interpolate. We apply the polarization rotation
	psi += phi, where phi is the longitude coordinate and psi is the polarization
	angle. This unwraps the polarization quadrupole we get around each pole.
	This rotation must be undone elsewhere before we can read off the true signal.
	Becase the polarization angle now is phi higher than it should be, the read-out
	code must add phi to its psi too."""
	theta, phi = field.map.posmap()
	map = field.map.copy()
	map[1:3] = enmap.rotate_pol(map[1:3], phi*np.sin(theta))
	#map[1:3] = enmap.rotate_pol(map[1:3], phi)
	return Field(field.name, map, field.spec, field.beam)

def calc_subfield(field, shape, wcs, target_beam, subsample=1, pad=0, apod=0):
	"""Given a field, compute its value on a new grid given by shape and wcs,
	smoothing it to the target_beam. The smoothing is done on a subsampled
	grid to reduce pixelization effects."""
	# Get our work geometry
	wshape, wwcs = subsample_geometry(shape, wcs, subsample)
	wshape, wwcs = pad_geometry(wshape, wwcs, pad/utils.degree)
	# Project onto it
	wfield = field.project(wshape, wwcs)
	# Smooth this field to the target beam
	wfield = wfield.to_beam(target_beam, apod=apod)
	# And interpolate to the target raster
	res = wfield.project(shape, wcs)
	return res

# It is often convenient to have spec and beam as plain
# arrays rather than functions. This makes transforming
# them easy. For example deriving the spectrogram from
# the spectrum or dividing by the beam. However, the
# spectrum isn't just proportional to the temperature.
# To be fast and memory-efficient, we want to be able to
# directly evaluate tcorr(pos, delay) without
# needing to do interpol(spec2tcorr(spec(pos, freqs)), delay).
# If spec was prop to T, then one could perform spec2tcorr once
# and then just scale the result. But it isn't.
#
# We may be able to find analytic expressions for the graybody
# autocorrelation function, though. Let's see:
#
# gray(T,b,f) = A*f**(3+b)/(exp(qf/T)-1)
# where A = (2h/c^2) and q = h/kb
# 
# gray(T,b,f) = gray(aT,b,af)/a**(3+b)
#
# so for any given b, we can precompute just a single
# interpolating function, and then just look up with scaling.
# And for graybody we just have a constant b, so this will work.


class Field:
	def __init__(self, name, map, spec, beam, order=3, pmap=None, copy=False):
		"""A field comprising one component of the sky. Specified
		by an enmap "map", a spectrum response spec[, and a beam
		function beam"""
		self.name = name
		self.spec = spec
		self.beam = beam
		self.order= order
		self.map  = map
		self.pmap = pmap
		if self.pmap is None:
			self.pmap = utils.interpol_prefilter(map, order=order)
		if copy:
			self.map  = self.map.copy()
			self.pmap = self.pmap.copy()
	def copy(self, deep=False):
		"""Perform a copy of the field"""
		return Field(self.name, self.map, self.spec, self.beam, order=self.order, pmap=self.pmap, copy=deep)
	def to_filter(self, filter):
		"""Return a new field which uses the given spectral filter"""
		res = self.copy()
		res.spec.set(filter=filter)
		return res
	def to_beam(self, beam):
		"""Return a new field with the given beam. Numerical problems may
		arise if the new beam is smaller than the old one."""
		map = apply_beam_fullsky(self.map, beam, self.beam)
		return Field(self.name, map, self.spec, beam, order=self.order)
	def at(self, freq, pos, type="spec"):
		"""Evaluate the signal at the given freq[{fdims}] and pos[2,{pdims}],
		returing res[{TQU},...]. fdims and pdims must be broadcastable
		to each other."""
		# amps will be [{TQU},pdims], so pdims and fdims must be broadcastable.
		amps = self.pmap.at(pos, order=self.order, prefilter=False, mask_nan=False, safe=False)
		return self.spec(freq, amps, type=type)

##### Spectrum types #####

class SpecBlackbody:
	type = "blackbody"
	def __init__(self, name, Tref, filter=None):
		self.set(name=name, Tref=Tref, filter=filter)
	def set(self, **kwargs):
		for key in kwargs: setattr(self, key, kwargs[key])
		if len(set(kwargs) & set(["Tref","filter"])) == 0: return
		self.core = SpecInterpol(blackbody, self.Tref, filter=self.filter)
	def __call__(self, freq, amps, type="spec"):
		"""Evaluate the frequency spectrum at the frequencies
		given by freq for the given blackbody amplitudes[{T,Q,U}].
		freq and amps[0] must be broadcastable. If type is "tcorr", then
		freq will actually be delays, and the result will be the
		time-correlation function."""
		# Make output array with broadcasting
		res = np.asanyarray(freq)+np.asanyarray(amps)
		fun = {
				"spec": self.core.eval_spec,
				"tcorr":self.core.eval_tcorr,
			}[type]
		res[0]  = fun(freq, amps[0])
		res[1:] = utils.tofinite(res[0] * amps[1:]/amps[0])
		return res

class SpecGraybody:
	"""This class implements a specialized graybody model appropriate
	for dust. The dust is all taken to be at a uniform temperature,
	and the T amplitude just represents differenting optical depth
	scaling."""
	type = "graybody"
	def __init__(self, name, T, beta, fref, unit, filter=None):
		self.set(name=name, T=T, beta=beta, fref=fref, unit=unit, filter=filter)
	def set(self, **kwargs):
		for key in kwargs: setattr(self, key, kwargs[key])
		if len(set(kwargs) & set(["T","filter","beta"])) == 0: return
		# Taylor is 0 because we will only call this with a fixed temperature
		# for now. May want to change this to support more general dust models
		# later.
		self.core = SpecInterpol(graybody, self.T, filter=self.filter, beta=self.beta, taylor=0)
		self.ref_scale = graybody(self.fref, self.T, self.beta)
	def __call__(self, freq, amps, type="spec"):
		"""Evaluate the frequency spectrum at the frequencies
		given by freq for the given graybody amplitudes[{T,Q,U}].
		freq and amps must be broadcastable. If type is "tcorr", then
		freq will actually be delays, and the result will be the
		time-correlation function."""
		fun = {
				"spec": self.core.eval_spec,
				"tcorr":self.core.eval_tcorr,
			}[type]
		return amps * fun(freq, self.T) / self.ref_scale * self.unit

##### Beam types #####

class Beam:
	type = "none"
	fwhm = 0
	def __call__(self, l):
		l = np.asarray(l)
		return np.full(l.shape, 1.0)
	def copy(self): return copy.deepcopy(self)

class BeamGauss(Beam):
	type = "gauss"
	def __init__(self, sigma):
		self.sigma = sigma
	def __call__(self, l):
		l = np.asarray(l)
		return np.exp(-0.5*l**2*self.sigma**2)

##### Filters #####

class Filter:
	def __call__(self, freqs): return np.full(freqs.shape, 1.0, freqs.dtype)

class FilterGauss:
	def __init__(self, sigma):
		self.sigma = sigma
	def __call__(self, freqs):
		return np.exp(-0.5*(freqs/self.sigma)**2)

class FilterButter:
	def __init__(self, fknee, alpha):
		"""Butterworth filter. Positive alpha gives lowpass filter,
		negative alpha gives highpass fitler."""
		self.fknee = fknee
		self.alpha = alpha
	def __call__(self, freqs):
		with utils.nowarn():
			profile = 1/(1+(freqs/self.fknee)**self.alpha)
		return profile

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
		return BeamGauss(params["fwhm"]*utils.degree*utils.fwhm)
	else:
		raise ValueError("Unknown beam type '%s'" % params["type"])

def parse_filter(params):
	if params["type"] is "none":
		return Filter()
	elif params["type"] is "gauss":
		return FilterGauss(params["sigma"])
	else:
		raise ValueError("Unknown filter type '%s'" % params["type"])

def parse_barrel(params):
	return [bunch.Bunch(
			sky      = sub["sky"],
			filter   = sub["filter"],
			beam     = sub["beam"],
			offset   = sub["offset"],
			response = sub["response"]
		) for sub in params]

def parse_det(params):
	return bunch.Bunch(horn=int(params["horn"]), response=np.array(params["response"]))

def parse_tod_filter(params):
	if params["type"] is "none":
		return Filter()
	elif params["type"] is "butter":
		return FilterButter(params["fknee"], params["alpha"])

def parse_ints(s): return parse_numbers(s, int)
def parse_floats(s): return parse_numbers(s, float)
def parse_numbers(s, dtype=None):
	res = []
	for word in s.split(","):
		toks = [float(w) for w in word.split(":")]
		if ":" not in word:
			res.append(toks[:1])
		else:
			start, stop = toks[:2]
			step = toks[2] if len(toks) > 2 else 1
			res.append(np.arange(start,stop,step))
	res = np.concatenate(res)
	if dtype is not None:
		res = res.astype(dtype)
	return res

##### Helpers #####
Tcmb= 2.725
h   = 6.626070040e-34
c   = 299792458
kb  = 1.38064853e-23

def rmul(R, a): return np.einsum("...ij,...jk->...ik",a,R)
def rmat(ax, ang): return utils.rotmatrix(ang, ax)
def rot(a, ax, ang): return rmul(rmat(ax,ang),a)
def add_patch(patches, isky, ibtype, patch):
	if patches[isky][ibtype] is None:
		patches[isky][ibtype] = patch
	else:
		patches[isky][ibtype].map += patch.map
def capped_ratio(a,b, cap=1):
	return np.minimum(a/b,cap)

def broadcast_stack(a, b):
	a = np.asanyarray(a)
	b = np.asanyarray(b)
	adim, bdim = a.ndim, b.ndim
	a = a[(Ellipsis,) + (None,)*bdim]
	b = b[(None,)*adim]
	return a, b

def blackbody(freqs, T, T_derivs=None):
	"""Given a set of frequencies freqs[{fdims}] and a set of
	temperatures T[{tdims}], returns a blackbody spectrum
	spec[{fdims},{tdims}]. All quantities are in SI units."""
	return graybody(freqs, T, 0, T_derivs=T_derivs)

def graybody(freqs, T, beta, T_derivs=None):
	"""Given a set of frequencies freqs[{fdims}] and a set of
	temperatures T[{tdims}] and emmisivities beta[{tdims}], returns a graybody spectrum
	spec[{fdims},{tdims}]. All quantities are in SI units.
	If T_derivs is not None, it should be a nonzero integer
	indicating the highest derivative with respect to T to include.
	For example, if T_derivs==2, the result will be (0th,1st,2nd)
	derivative. In this case, the result will have dimension
	[T_derivs+1,{fdims},{tdims}]."""
	freqs, T = broadcast_stack(freqs, T)
	beta     = np.asanyarray(beta)
	norder   = 1
	if T_derivs is not None: norder = T_derivs + 1
	res = np.zeros((norder,)+np.broadcast(freqs,T).shape)
	# Common factor for all orders
	pre= (2*h/c**2) * freqs**(3+beta)
	with utils.nowarn():
		# Building blocks of derivatives
		a  = h*freqs/kb
		g0 = np.exp(a/T)
		f0 = 1/(g0-1)
		# 0th order
		if norder > 0:
			res[0] = f0
		if norder > 1:
			g1 = -g0*a/T**2
			f1 = -f0**2*g1
			res[1] = f1
		if norder > 2:
			g2 = a/T**3*(2*g0-T*g1)
			f2 = -2*f0*f1*g1-f0**2*g2
			res[2] = f2
		if norder > 3:
			g3 = -(3./T+a/T**2)*g2 + a/T**3*g1
			f3 = -2*f1**2*g1-2*f0*f2*g1-4*f0*f1*g2-f0**2*g3
			res[3] = f3
		if norder > 4:
			raise NotImplementedError
		res *= pre
	res  = utils.tofinite(res)
	# Don't include derivative dimension if derivatives were disabled.
	if T_derivs is None:
		res = res[0]
	return res

class InterpolatedTaylorLookup:
	def __init__(self, coeffs, pstep, interpol=3, x0=0):
		"""Construct an object that can be used to approximate some function
		by using the coefficients of its taylor expansion, where these coefficients
		are a function of position and are regularly sampled with interval pstep.
		coeffs is an array [order+1,npos], where the corresponding positions are
		[0,1*pstep,2*pstep,...]. The optional argument interpol specifies the
		order of the interpolation to be used when looking up the taylor expansion
		coefficients given a position. This is not the same thing as the order
		to be used in the taylor expansion, which is given by the number of
		coefficients passed in."""
		self.coeffs = coeffs
		self.coeffs_pre = [utils.interpol_prefilter(coeff, npre=0, order=interpol) for coeff in coeffs]
		self.pstep    = pstep
		self.interpol = interpol
		self.x0       = x0
	def __call__(self, pos, x, order=None):
		"""Evaluate the function at positions pos[:] for the function
		value x using all taylor coefficients, or only up to that order
		specified by the optional order argument."""
		if order is None: order = len(self.coeffs_pre)-1
		pix = np.asarray(pos)/self.pstep
		dx  = x-self.x0
		res, fac = None, 1.0
		for i in range(order+1):
			c = utils.interpol(self.coeffs_pre[i], pix[None], order=self.interpol, prefilter=False)
			if res is None: res = c
			else: res += fac*dx**i*c
			fac /= (i+1)
		return res

class SpecInterpol:
	"""This class allows fast lookups of spectrum and spectrogram (time
	correlation function) values."""
	def __init__(self, specfun, Tref, filter=None, nsamp=50000, order=3, taylor=3, fmax=None, **kwargs):
		"""specfun(freqs, T, T_derivs=num, **kwargs):
		  A function implementing the spectrum to be interpolated.
		  Returns spectrum[num,nfreq]
		Tref: The temperature to Taylor-expand the spectrum around. The spectrum
		  is assumed to be a nonlinear function of the temperature.
		filter(freqs): An optional frequency response function, which the spectrum
		  is multiplied by.
		nsamp [int]: The number of samples to use in the interpolation
		order [int]: The spline order to use in the interpolation
		taylor [int]: The order of the Taylor expansion
		fmax [float]: The highest frequency to use in the interpolation. Should
		  be high enough to encompass the whole spectrum.
		Any additional arguments are passed on to specfun."""
		if fmax is None: fmax = kb*Tref/h*100
		fwcs  = freq_geometry(fmax, nsamp)
		freqs = fwcs.wcs_pix2world(np.arange(nsamp),0)[0]
		spec_series = specfun(freqs, Tref, T_derivs=taylor, **kwargs)
		# Apply filter if necessary
		if filter is not None:
			filter_vals = filter(freqs)
			for spec in spec_series:
				spec *= filter_vals
		# Transform each order of the Taylor series
		tcorr_series = []
		for spec in spec_series:
			tcorr, twcs = spec2delay(spec, fwcs)
			tcorr_series.append(tcorr)
		# Build our Taylor lookups
		self.tcorr_lookup = InterpolatedTaylorLookup(tcorr_series, twcs.wcs.cdelt[0], interpol=order, x0=Tref)
		self.spec_lookup  = InterpolatedTaylorLookup(spec_series,  fwcs.wcs.cdelt[0], interpol=order, x0=Tref)
		self.filter = filter
	def eval_spec (self, freq,  T, order=None):
		return self.spec_lookup (freq, T, order=order)
	def eval_tcorr(self, delay, T, order=None):
		return self.tcorr_lookup(delay, T, order=order)

# These are superceded by SpecInterpol. They use a different philosophy
# that lets them cover all temperatures, not just those close to a reference
# point, at the cost of not being able to support a filter.
def blackbody_tcorr(xmax=100, nsamp=50000, order=3):
	return graybody_tcorr(beta=0, xmax=xmax, nsamp=nsamp, order=3)
class graybody_tcorr:
	"""This class allows direct evaluation of the time_correlation function
	corresponding to a graybody with a fixed beta parameter."""
	def __init__(self, beta=0, xmax=100, nsamp=50000, order=3):
		self.beta, self.order, self.Tref = beta, order, h/kb
		# Build spec at reference temperature
		fwcs  = freq_geometry(xmax, nsamp)
		freqs = fwcs.wcs_pix2world(np.arange(nsamp),0)[0]
		spec  = graybody(freqs, self.Tref, self.beta)
		# Get corresponding tcorr
		tcorr, twcs = spec2delay(spec, fwcs)
		# Prefilter and extract scale
		self.pcorr = utils.interpol_prefilter(tcorr, npre=0, order=self.order)
		self.tstep = twcs.wcs.cdelt[0]
	def __call__(self, delay, T):
		a = T/self.Tref
		return utils.interpol(self.pcorr, np.asarray(delay)[None]*a/self.tstep, order=self.order, prefilter=False)*a**(1+3+self.beta)

def spec2delay(arr, wcs, axis=0, inplace=False, bsize=32):
	"""Converts a spectrum cube arr[nfreq,...] into an
	autocorrelation function [delay,...]. The frequency
	information is described by the spectral wcs argument.
	Returns corrfun, delay_wcs. If the input units are
	W/sr/m^2/Hz, then the output units will be
	W/sr/m^2."""
	arr = np.asanyarray(arr)
	if not inplace: arr = arr.copy()
	with utils.flatview(arr, [axis], "rw") as aflat:
		n = aflat.shape[0]
		nfreq = aflat.shape[1]
		nb = (n+bsize-1)/bsize
		for bi in range(nb):
			i1, i2 = bi*bsize, min(n,(bi+1)*bsize)
			aflat[i1:i2] = fft.redft00(aflat[i1:i2]) * 0.5
	owcs = wcs_spec2delay(wcs, nfreq)
	# Take into account the units on the x-axis.
	arr *= wcs.wcs.cdelt[0]
	return arr, owcs

def delay2spec(arr, wcs, axis=0, inplace=False, bsize=32):
	"""Converts an autocorrelation cube arr[ndelay,...] into an
	autocorrelation function [delay,...]. The delay
	information is described by the temporal wcs argument.
	If the input units are W/sr/m^2, then the output units will
	be W/sr/m^2/Hz."""
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
	owcs = wcs_delay2spec(wcs, ndelay)
	arr /= owcs.wcs.cdelt[0]
	return arr, owcs

def wcs_delay2spec(wcs, ndelay):
	owcs = wcs.deepcopy()
	owcs.wcs.cdelt[0] = c/wcs.wcs.cdelt[0]/ndelay/2
	owcs.wcs.ctype[0] = 'FREQ'
	return owcs

def wcs_spec2delay(wcs, nfreq):
	owcs = wcs.deepcopy()
	owcs.wcs.cdelt[0] = c/wcs.wcs.cdelt[0]/nfreq/2
	owcs.wcs.ctype[0] = 'TIME'
	return owcs

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
	wcs.wcs.cdelt[0] = float(fmax)/nfreq
	wcs.wcs.crpix[0] = 1
	wcs.wcs.ctype[0] = 'FREQ'
	return wcs

def longitude_geometry(box, res=0.1*utils.degree, dims=(), ref=None):
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
	if ref is not None:
		# It may be useful to have ra0 and dec0 at an integer number of
		# unrotated CAR pixels from some reference point
		dec0 = np.round((dec0 - ref[0])/res)*res + ref[0]
		ra0  = np.round((ra0  - ref[1])/res)*res + ref[1]
		wdec = np.ceil(np.abs(wdec)/(2*res))*2*res*np.sign(wdec)
		wra  = np.ceil(np.abs(wra) /(2*res))*2*res*np.sign(wra)
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
		wcs.wcs.lonpole =  90
		# Transform the points. Since cdelt = 1, these new
		# pixel coordinates will correspond to flat-sky angles
		x, y = wcs.wcs_world2pix(point[0], point[1], 0)
		# We want to be symmetric around phiref
		phiwidth = max(np.abs(y))
		box = np.array([
			[thetaref+np.min(x),phiref-phiwidth],
			[thetaref+np.max(x),phiref+phiwidth]])
		phiref -= np.mean(y)
	return box*utils.degree

def subsample_geometry(shape, wcs, nsub):
	owcs = wcs.deepcopy()
	#owcs.wcs.crpix -= 0.5
	owcs.wcs.crpix *= nsub
	owcs.wcs.cdelt /= nsub
	#owcs.wcs.crpix += 0.5
	oshape = tuple(shape[:-2]) + tuple(np.array(shape[-2:])*nsub)
	return oshape, owcs

def pad_geometry(shape, wcs, pad):
	owcs = wcs.deepcopy()
	pad_pix = (np.abs(pad/owcs.wcs.cdelt)/2).astype(int)
	owcs.wcs.crpix += pad_pix
	oshape = tuple(shape[:-2]) + tuple(np.array(shape[-2:])+2*pad_pix)
	return oshape, owcs

def overlap_cat(arrays, axis=-1, n=1):
	"""Given a list of arrays [:][...,nsamp], where each array has n overlap with the
	previous. Return the concatenation of the arrays along the last dimension after
	discarding overlapping samples and adding offsets needed to make those samples match."""
	work = [arrays[0]]
	for array in arrays[1:]:
		a = utils.moveaxis(work[-1],axis,-1)
		b = utils.moveaxis(array,   axis,-1)
		off = np.mean(b[...,:n]-a[...,-n:],-1)
		c = b[...,n:] - off[...,None]
		work.append(utils.moveaxis(c, -1, axis))
	return np.concatenate(work,axis)

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

def load_config(path=None):
	"""Load the configuration at the given path, or the at
	the same location as pixie.py if none is specified. Returns
	the configuration object."""
	if path:
		config = imp.load_source("config", path)
	else:
		import config
	return config

def froll(a, shift, axis=-1, noft=False, inplace=False):
	"""Roll array elements along the given axis. Unlike np.roll,
	the shift can be fractional. Fourier shifting is used for this.
	If noft is True, the input and output arrays will be assumed to
	already be in fourier domain. Otherwise, the fourier transform
	is performed internally."""
	if not inplace: a = np.asanyarray(a).copy()
	axis %= a.ndim
	fa    = a if noft else fft.fft(a, axes=(axis,))
	k     = np.fft.fftfreq(fa.shape[axis])
	# Expand indices so we broadcast correctly
	k     = k[(None,)*(axis)+(slice(None),)+(None,)*(fa.ndim-axis-1)]
	phase = np.exp(-2j*np.pi*k*shift)
	fa   *= phase
	if noft:
		a = fa
	else:
		tmp = fft.ifft(fa, axes=(axis,), normalize=True)
		a   = tmp if np.iscomplexobj(a) else tmp.real
	return a

def fix_drift(d):
	"""Given d[...,nt,nspin,ndelay], where d represents data changing
	smoothingly both as a function time and spin in 1 dimension (e.g.
	between one step in the last axis both t and spin change slightly),
	shift the array such that the axes become independent in the sense
	taht t and spin only change when one their respective index changes."""
	# Adjust phase to compensate for sky motion during a spin
	nt,nspin,ndelay = d.shape[-3:]
	d = froll(
			d.reshape(-1,nt,nspin*ndelay),
			np.arange(nspin*ndelay)/float(nspin*ndelay),
			-2).reshape(-1,nt,nspin,ndelay)
	## Adjust phase to compensate for spin during strokes
	d  = froll(d, np.arange(ndelay)[(None,)*(d.ndim-1)+(slice(None),)]/float(ndelay),-2)
	return d

def rot_comps(d, ang, axis=0, off=None):
	if off is None: off = d.shape[axis]-2
	c = np.cos(ang)
	s = np.sin(ang)
	res = d.copy()
	pre = (slice(None),)*axis
	res[pre+(off+0,)] = d[pre+(off+0,)] * c - d[pre+(off+1,)] * s
	res[pre+(off+1,)] = d[pre+(off+0,)] * s + d[pre+(off+1,)] * c
	return res
