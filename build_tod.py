import numpy as np, argparse, pixutils, h5py
from enlib import enmap, bunch, utils, bench
parser = argparse.ArgumentParser()
parser.add_argument("sky_map")
parser.add_argument("reference")
parser.add_argument("ofile")
parser.add_argument("-B", "--barrels",   type=str,   default="oo")
parser.add_argument("-i", "--start",     type=float, default=0)
parser.add_argument("-d", "--duration",  type=float, default=1)
parser.add_argument("-s", "--step",      type=int,   default=1)
parser.add_argument("-O", "--oversample",type=int,   default=4)
parser.add_argument("-I", "--interpol",  type=int,   default=3,
	help="""Interpolation order to use when looking up samples in the map. Orders
	higher than 1 (linear) require a slow one-time prefiltering of the map.
	If the order specified is negative, the input map assumes to have been already
	appropriately prefiltered, and the prefiltering is skipped.""")
parser.add_argument("--interpol-mode",   type=str,   default="plain")
args = parser.parse_args()

def rmul(R, a): return np.einsum("...ij,...jk->...ik",a,R)
def rmat(ax, ang): return utils.rotmatrix(ang, ax)
def rot(a, ax, ang): return rmul(rmat(ax,ang),a)
#def norm(a): return a/np.sum(a**2,-1)[...,None]**0.5

# FIXME: Should go to 7.4 THz (actually 512/8 of the CO line freq)

class ScanGenerator:
	def __init__(self):
		# Pointing of each barrel (bucket) relative to the barrel spin axis, in the
		# form of zyz euler angles
		self.barrel_offs = np.array([[0,0,0],[0,0,0]],dtype=float)
		# Responsivity of each detector in each barrel to horizontal and vertical pol
		self.det_angle   = np.array([0,np.pi/2,0,np.pi/2])
		self.det_barrels = np.array([ [0,1],[0,1],[1,0],[1,0] ])
		# Angle of the barrel spin axis relative to scan spin axis
		self.opening_angle=np.pi/2
		# Target optical delay at max stroke in mm
		self.delay_amp     = 0.01
		self.delay_period  = 8.0
		self.delay_phase   = 0.0
		self.spin_period   = 60.0
		self.spin_phase    = 0.0
		self.scan_period   = 384*self.spin_period
		self.scan_phase    = np.pi/2
		self.orbit_period  = 365.25636*24*3600
		self.orbit_phase   = 0.0
		self.eclip_ang     = 0.0
		# Orbit phase updates in steps of orbit_step seconds
		self.orbit_step    = self.scan_period
		self.ref_ctime     = 1500000000
		self.sample_period = 1/256.0
		self.nbarrel = len(self.barrel_offs)
	def gen_orbit(self, i0=0, n=100000, step=1, oversample=1, interpol_mode="plain"):
		"""Generate our orbital positions. These are needed to compute the pointing."""
		i_base  = np.arange(n)*step + i0
		if oversample == 1:
			off, weights = np.array([0.0]), np.array([1.0])
		else:
			if interpol_mode == "gauss":
				off, weights = np.polynomial.legendre.leggauss(oversample)
				# Go from [-1,1] to [-0.5,0.5]
				off /= 2
				weights /= 2
			elif interpol_mode == "plain":
				off = 0.5*((2*np.arange(oversample)+1)/float(oversample)-1)
				weights = np.full(oversample,1.0)/float(oversample)
			elif interpol_mode == "trap":
				off = np.arange(oversample)/float(oversample-1)-0.5
				weights = np.concatenate([[1],np.full(oversample-2,2.0),[1]])/(2.0*(oversample-1))
			elif interpol_mode == "simpson":
				oversample = oversample/2*2+1
				off = np.arange(oversample)/float(oversample-1)-0.5
				weights = np.concatenate([[1],((1+np.arange(oversample-2))%2+1)*2,[1]])/(3.0*(oversample-1))
				print off, weights
		t = (i_base[:,None] + off[None,:]).reshape(-1) * self.sample_period
		ang_orbit = self.orbit_phase   + 2*np.pi*np.floor(t/self.orbit_step)*self.orbit_step/self.orbit_period
		ang_scan  = self.scan_phase    + 2*np.pi*t/self.scan_period
		ang_spin  = self.spin_phase    + 2*np.pi*t/self.spin_period
		ang_delay = self.delay_phase   + 2*np.pi*t/self.delay_period
		return bunch.Bunch(t=t, orbit=ang_orbit, scan=ang_scan, spin=ang_spin, delay=ang_delay, weights=weights)
	def gen_pointing(self, oparam):
		"""Use orbital positions into pointing and orientation on the sky.
		Returns a bunch(point[{phi,theta},nbarrel,ntime], gamma[nbarrel,ntime], delay[ntime])
		"""
		# First set up the part of the pointing matrix that is common for all the
		# detectors. This consists of the rotation from the spin axis, to scan,
		# to orbit, but does not include the barrel.
		Ro = np.eye(3)
		Ro = rot(Ro, "z", oparam.spin)
		Ro = rot(Ro, "y", self.opening_angle)
		Ro = rot(Ro, "z", oparam.scan)
		Ro = rot(Ro, "y", np.pi/2 - self.eclip_ang)
		Ro = rot(Ro, "z", oparam.orbit)
		# Then set up the barrels
		Rb = np.eye(3)
		Rb = rot(Rb, "z", self.barrel_offs[:,0])
		Rb = rot(Rb, "y", self.barrel_offs[:,1])
		Rb = rot(Rb, "z", self.barrel_offs[:,2])
		# Combine to form the total rotation matrix
		R = np.einsum("tij,hjk->thik", Ro, Rb)
		# Switch to t-last ordering, as it is easier to work with
		R = np.einsum("thik->kiht", R)
		# Get the pointing for each barrel
		xvec, zvec = R[:,0], R[:,2]
		point = utils.rect2ang(zvec,axis=0, zenith=False)
		# Make sure phi is between -180 and 180
		point[0] = utils.rewind(point[0])
		# Get the polarization orientation on the sky
		gamma = np.arctan2(xvec[2], -zvec[1]*xvec[0]+zvec[0]*xvec[1])
		# And the delay at each time
		delay = self.delay_amp * np.sin(oparam.delay)
		return bunch.Bunch(point=point, gamma=gamma, delay=delay, pos=zvec)
	def gen_det_resp(self, gamma):
		"""Compute the on-sky T,Q,U sensitivity of each detector for each barrel.
		Will be [ndet,{0,delay},nbarrel,{T,Q,U},t] because every detector gets contributions from
		every barrel."""
		# Power incident on x and y-oriented detectors in the A and B horns.
		# Pax = 1/4*(Ia+Ib+Qa-Qb)[0] + 1/4*(Ia-Ib+Qa+Qb)[delta]
		# Pay = 1/4*(Ia+Ib-Qa+Qb)[0] + 1/4*(Ia-Ib-Qa-Qb)[delta]
		# Pbx = 1/4*(Ib+Ia+Qb-Qa)[0] + 1/4*(Ib-Ia+Qb+Qa)[delta]
		# Pby = 1/4*(Ib+Ia-Qb+Qa)[0] + 1/4*(Ib-Ia-Qb-Qa)[delta]
		# where the number in [] indicates the one-sided time-delay,
		# which is half the total path time difference.
		#
		# Overall, a detector in horn i with polarization angle alpha,
		# and with a sky rotation gamma for barrel i, will measure
		# P = 1/4*(Ii + Ij + Qi ci + Ui si - Qj cj - Uj sj)[0] +
		#     1/4*(Ii - Ij + Qi ci + Ui si + Qj cj + Uj sj)[delta]
		# where ci = cos(2*(alpha + gamma_i)), and so on.
		ndet    = self.det_angle.size
		nbarrel, ntime = gamma.shape
		res = np.zeros([ndet, 2, nbarrel, 3, ntime])
		for di in range(ndet):
			barrels = self.det_barrels[di]
			ang1 = 2*(gamma[barrels[0]] + self.det_angle[di])
			ang2 = 2*(gamma[barrels[1]] + self.det_angle[di])
			u = np.full(ntime, 1.0)
			c1, c2 = np.cos(ang1), np.cos(ang2)
			s1, s2 = np.sin(ang1), np.sin(ang2)
			res[di,:,:,:] = np.array([
				[[ u, c1, s1],[ u,-c2,-s2]],   # DC
				[[ u, c1, s1],[-u, c2, s2]]])  # delay
		return res
	def gen_barrel_pixels(self, bpoint, delay, wcs_pos, wcs_delay):
		"""Maps pointing to pixels[{pix_dc, pix_delay, pix_y, pix_x},ntime]
		for the specified barrel. These are used in gen_signal_barrel to
		look up the sky signal for each sample."""
		ntime = bpoint.shape[-1]
		bpix = np.zeros([4,ntime])
		bpix[0]  = wcs_delay.wcs_world2pix([0], 1)[0]
		bpix[1]  = wcs_delay.wcs_world2pix(np.abs(delay), 1)[0]
		# bpoint is [{phi,theta},ntime], but world2pix wants [:,{phi,theta}] in degrees
		pdeg = bpoint.T / utils.degree
		# The result will be [:,{x,y}], but we want [{y,x},ntime]
		bpix[3:1:-1] = wcs_pos.wcs_world2pix(pdeg,0).T
		return bpix
	def gen_barrel_signal(self, bsky, bpix, bresp, order=3):
		"""Generates a simulated sky signal for a *single barrel* by evaluating
		the given sky mal at the given positions with each detectors response using interpolation
		of the given order. Returns signal[ndet,ntime]."""
		# First evaluate the sky at each barrels position, both for our delay and 0 delay.
		# skies is a list of sky arrays, one per barrel. Each sky is [{T,Q,U},ndelay,y,x]
		ndet, ntime = bresp.shape[0], bresp.shape[-1]
		# Since sky is [{T,Q,U},ndelay,y,x], sig_barrel will be [{T,Q,U},nbarrel,ntime]
		sig_barrel_dc    = utils.interpol(bsky, bpix[[0,2,3]], order=order, mode="constant", mask_nan=False, prefilter=False)
		sig_barrel_delay = utils.interpol(bsky, bpix[1:],      order=order, mode="constant", mask_nan=False, prefilter=False)
		# resp[ndet,{0,delay},nbarrel,{T,Q,U},ntime] tells us how to turn the
		# barrel signals into detector outputs
		bsignal  = np.einsum("dct,ct->dt", bresp[:,0], sig_barrel_dc)
		bsignal += np.einsum("dct,ct->dt", bresp[:,1], sig_barrel_delay)
		return bsignal

ofile = args.ofile
order = args.interpol
with bench.show("read"):
	# Read our sky cube, which should be [{T,Q,U},ndelay,y,x]
	sky, spec_wcs = pixutils.read_map(args.sky_map)
	ref, ref_wcs  = pixutils.read_map(args.reference)
with bench.show("prefilter"):
	# Prefilter to make interpolation faster
	if order > 1: sky = utils.interpol_prefilter(sky, order)
	if order > 1: ref = utils.interpol_prefilter(ref, order)
# Set up our skies
skies = [(sky,spec_wcs) if state == "o" else (ref,ref_wcs) for state in args.barrels]

sgen = ScanGenerator()
with bench.show("orbit"):
	offset = int(sgen.scan_period/sgen.sample_period * args.start)
	num    = int(sgen.scan_period/sgen.sample_period * args.duration)
	oparam = sgen.gen_orbit(i0=offset, n=num, step=args.step, oversample=args.oversample, interpol_mode=args.interpol_mode)
with bench.show("pointing"):
	pinfo  = sgen.gen_pointing(oparam)
with bench.show("response"):
	resp   = sgen.gen_det_resp(pinfo.gamma)
with bench.show("signal"):
	signal = np.zeros([resp.shape[0], resp.shape[-1]])
	for barrel in range(sgen.nbarrel):
		bsky, bspec_wcs = skies[barrel]
		bpix    = sgen.gen_barrel_pixels(pinfo.point[:,barrel], pinfo.delay, bsky.wcs, bspec_wcs)
		signal += sgen.gen_barrel_signal(bsky, bpix, resp[:,:,barrel], order=np.abs(order))
with bench.show("downgrade"):
	nsub    = len(oparam.weights)
	osignal = signal.reshape(signal.shape[:-1] + (signal.shape[-1]/nsub, nsub))
	osignal = np.sum(osignal*oparam.weights,-1)

with bench.show("write"):
	with h5py.File(ofile,"w") as hfile:
		hfile["point"] = pinfo.point
		hfile["pos"]   = pinfo.pos
		hfile["gamma"] = pinfo.gamma
		hfile["tod"]   = osignal
		hfile["tod_highres"] = signal
		hfile["pix"]   = bpix
