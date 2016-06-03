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
orbit_period  = 365.25636*24*3600
orbit_phase   = 0
orbit_step    = scan_period
eclip_angle   = 0
# The optical delay due to the mirror stroke
delay_period  = spin_period/8.0
delay_amp     = 0.01
delay_phase   = 0

# Frequencies
fmax          = 7.4e12
nfreq         = 2048
lmax          = 5000
nl            = 1000

# The optical filter
filter_method = "scatter"
filter_freq   = 1.5e12

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
	{"sky": 0, "subbeams": [ {"type": 0, "offset": [0,0,0], "response": I} ] },
]

# The beams
beam_types = [
	# Standard frequency-independent Gaussian
	{"type": "gauss", "fwhm": 1.9}
]

# The skies
skies = [
	# The actual sky
	[ "imaps/map_cmb.fits", "imaps/map_dust.fits" ],
	# The reference blackbody
	[ "imaps/map_ref.fits" ],
]
