# Unix time for sample 0
ref_ctime     = 1500000000
sample_period = 1./256
# The spin about the axis
spin_period   = 60
spin_phase    = 0
opening_angle = 0
# The great-circle scans
scan_period   = 384*spin_period
scan_phase    = 90
# And the solar orbit
#orbit_period  = 365.25636*24*3600
# Reduced orbital period to make sims less oversampled
orbit_period  = scan_period*scan_period/spin_period
orbit_phase   = 90
orbit_step    = scan_period
orbit_step_dur= 1
eclip_angle   = 0
# The optical delay due to the mirror stroke
delay_shape   = "triangle"
delay_period  = spin_period/8.0
delay_amp     = 0.01040303
delay_phase   = 0

# Frequencies
fmax          = 7377.3568e9
nfreq         = 2048
lmax          = 5000
nl            = 1000

# Time-domain filter
tod_filters = [
		{"type":"butter", "fknee":0.01, "alpha":-5},
		{"type":"butter", "fknee":100,  "alpha":+5}
	]

# These affect memory use and accuracy
subsample_num = 7
subsample_method = "gauss"

# These affect sub-patches used in the spectrogram
# calculation.
chunk_size    = 3e5
bounds_skip   = 13
bounds_niter  = 5
# Sensitive to beam_nsigma even when not doing any smoothing
# or apodization at that stage. Going from 3 to 4 took my
# residuals from 1.0e-6 to 3.6e-6.
beam_nsigma   = 6
patch_res     = 0.2
patch_nsub    = 5
patch_pad     = 10
patch_apod    = 0.6

# The detectors
dets = [
	{"horn": 0, "response": [ 1, 1, 0]},
	{"horn": 0, "response": [ 1,-1, 0]},
	{"horn": 1, "response": [ 1, 1, 0]},
	{"horn": 1, "response": [ 1,-1, 0]},
]

# The barrels
I = [[1,0,0],[0,1,0],[0,0,1]]
barrels = [
	{"sky": 0, "subbeams": [ {"type": 0, "offset": [0,0,0], "response": I} ] },
	{"sky": 1, "subbeams": [ {"type": 0, "offset": [0,0,0], "response": I} ] },
]

# The beams
beam_types = [
	# Standard frequency-independent Gaussian
	{"type": "gauss_scatter", "fwhm": 1.9, "freq_cut": 1.5e12}
]

# The skies
skies = [
	# The actual sky
	[ "imaps/map_cmb.fits", "imaps/map_dust.fits" ],
	# The reference blackbody
	[ "imaps/map_ref.fits" ],
]
