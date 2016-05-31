# Unix time for sample 0
ref_ctime     = 1500000000
sample_period = 1./256
# The optical delay due to the mirror stroke
delay_period  = 8
delay_amp     = 0.01
delay_phase   = 0
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

# Frequencies
fmax          = 7.4e12
nfreq         = 2048
lmax          = 500
nl            = 500

# These affect memory use and accuracy
subsample_num = 5
subsample_method = "gauss"

# These affect sub-patches used in the spectrogram
# calculation.
chunk_size    = 4e5
bounds_skip   = 13
bounds_niter  = 7
beam_nsigma   = 7
patch_res     = 0.25

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
