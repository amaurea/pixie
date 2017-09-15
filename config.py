import numpy as np
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
orbit_step    = scan_period
orbit_step_dur= 1
orbit_period  = scan_period*scan_period/spin_period
orbit_phase   = 90
eclip_angle   = 0
# The optical delay due to the mirror stroke
delay_shape   = "triangle"
delay_period  = spin_period/8.0
delay_phase   = 0
delay_amp     = 0.01040303
delay_jitter  = 0e-6
jitter_model  = "waves"

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
no_dc = False

# These affect memory use and accuracy
subsample_num    = 9
subsample_method = "gauss"
chunk_size       = 1e4

# This is a more compact representation of the detector signals. The
# signal part has been separated into a separate array of filter-beam-offset-response
# combinations. The "signal" property gives the element from the signal
# array to use for each barrel. The white noise in a detector has variance
# per second of sigma**2.
dets = [
	{"horn":0, "signal":[0,0], "response":[ 1, 1, 0], "sigma":0e-15, "fknee":0.1, "alpha":3},
	{"horn":0, "signal":[0,0], "response":[ 1,-1, 0], "sigma":0e-15, "fknee":0.1, "alpha":3},
	{"horn":1, "signal":[0,0], "response":[ 1, 1, 0], "sigma":0e-15, "fknee":0.1, "alpha":3},
	{"horn":1, "signal":[0,0], "response":[ 1,-1, 0], "sigma":0e-15, "fknee":0.1, "alpha":3},
]

I = [[1,0,0],[0,1,0],[0,0,1]]
leak = 1e-1
L = [[1-leak,leak,0],[leak,1-leak,0],[0,0,1]]
off = 1*np.pi/180/60/60
signals = [
		[ {"filter": 0, "beam": 0, "offset": [0,0,0], "response": I} ],
		[ {"filter": 0, "beam": 0, "offset": [0,0,0], "response": L} ],
		[ {"filter": 0, "beam": 0, "offset": [0,off,0], "response": I} ],
	]
# Which sky each barrel sees
barrels = [ {"sky":0}, {"sky":1} ]

# The beams
beams = [
	{"type": "gauss", "fwhm": 1.9}
]
filters = [
	{"type": "gauss", "sigma": 1.5e12/2**0.5}
]

# Sometimes a single, representative value is needed
fiducial_beam   = 0
fiducial_filter = 0

# The skies
skies = [
	#[ "imaps/map_sgrid.fits" ],
	#[ "imaps/map_ref_test.fits" ],
	# The actual sky
	[ "imaps/map_cmb.fits", "imaps/map_dust.fits" ],
	# The reference blackbody
	[ "imaps/map_ref.fits" ],
]
