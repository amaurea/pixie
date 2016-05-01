import numpy as np, argparse, pixutils, h5py
from enlib import enmap, bunch, utils, bench
parser = argparse.ArgumentParser()
parser.add_argument("sky_map")
parser.add_argument("ofile")
parser.add_argument("-i", "--scan",     type=float, default=0)
parser.add_argument("-b", "--nscan",    type=float, default=1)
parser.add_argument("-s", "--step",     type=int,   default=1)
parser.add_argument("-I", "--interpol", type=int, default=3,
	help="""Interpolation order to use when looking up samples in the map. Orders
	higher than 1 (linear) require a slow one-time prefiltering of the map.
	If the order specified is negative, the input map assumes to have been already
	appropriately prefiltered, and the prefiltering is skipped.""")
args = parser.parse_args()

def rmul(R, a): return np.einsum("...ij,...jk->...ik",a,R)
def rmat(ax, ang): return utils.rotmatrix(ang, ax)
def rot(a, ax, ang): return rmul(rmat(ax,ang),a)
#def norm(a): return a/np.sum(a**2,-1)[...,None]**0.5

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
		self.delay_period  = 1.0
		self.delay_phase   = 0.0
		self.spin_period   = 60.0
		self.spin_phase    = 0.0
		self.scan_period   = 512*self.spin_period
		self.scan_phase    = np.pi/2
		self.orbit_period  = 365.25636*24*3600
		self.orbit_phase   = 0.0
		self.eclip_ang     = 0.0
		# Orbit phase updates in steps of orbit_step seconds
		self.orbit_step    = self.scan_period
		self.ref_ctime     = 1500000000
		self.sample_period = 1/256.0
	def gen_orbit(self, i0=0, n=100000, step=1):
		"""Generate our orbital positions. These are needed to compute the pointing."""
		t = (np.arange(n)*step + i0)*self.sample_period
		ang_orbit = self.orbit_phase   + 2*np.pi*np.floor(t/self.orbit_step)*self.orbit_step/self.orbit_period
		ang_scan  = self.scan_phase    + 2*np.pi*t/self.scan_period
		ang_spin  = self.spin_phase    + 2*np.pi*t/self.spin_period
		ang_delay = self.delay_phase   + 2*np.pi*t/self.delay_period
		return bunch.Bunch(t=t, orbit=ang_orbit, scan=ang_scan, spin=ang_spin, delay=ang_delay)
	def gen_pointing(self, oparam):
		"""Use orbital positions into pointing and orientation on the sky.
		Returns a bunch(point[{phi,theta},nhorn,ntime], gamma[nhorn,ntime], delay[ntime])
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
	def gen_det_resp(self, point):
		"""Compute the on-sky T,Q,U sensitivity of each detector for each pointing.
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
		nbarrel, ntime = point.gamma.shape
		res = np.zeros([ndet, 2, 2, 3, ntime])
		for di in range(ndet):
			barrels = self.det_barrels[di]
			ang1 = 2*(point.gamma[barrels[0]] + self.det_angle[di])
			ang2 = 2*(point.gamma[barrels[1]] + self.det_angle[di])
			u = np.full(ntime, 1.0)
			c1, c2 = np.cos(ang1), np.cos(ang2)
			s1, s2 = np.sin(ang1), np.sin(ang2)
			res[di,:,:,:] = np.array([
				[[ u, c1, s1],[ u,-c2,-s2]],
				[[ u, c1, s1],[-u, c2, s2]]])
		return res
	def gen_pixels(self, point, wcs_pos, wcs_delay):
		"""Maps pointing to pixels[{pix_delay, pix_delay_dc, pix_y, pix_x}].
		These are used in gen_signal to look up the sky signal for each sample."""
		nbarrel, _, ntime = point.point.shape
		pix = np.zeros([4,nbarrel,ntime])
		pdeg = point.point / utils.degree
		pix[0]  = wcs_delay.wcs_world2pix(np.abs(point.delay), 1)[0]
		pix[1]  = wcs_delay.wcs_world2pix([0], 1)[0]
		pix[3:1:-1] = np.array(wcs_pos.wcs_world2pix(pdeg.T.reshape(-1,2), 0)).T.reshape(point.point.shape)
		return pix
	def gen_signal(self, sky, pix, resp, order=3):
		"""Generates a simulated sky singla by evaluating the given sky map
		at the given positions with each detectors response using interpolation
		of the given order. Returns signal[ndet,ntime]."""
		# First evaluate the sky at each barrels position, both for our delay and 0 delay.
		# Since sky is [{T,Q,U},ndelay,y,x], sig_barrel will be [{T,Q,U},nbarrel,ntime]
		sig_barrel_dc    = utils.interpol(sky, pix[1:],      order=order, mode="constant", mask_nan=False, prefilter=False)
		sig_barrel_delay = utils.interpol(sky, pix[[0,2,3]], order=order, mode="constant", mask_nan=False, prefilter=False)
		# resp[ndet,{0,delay},nbarrel,{T,Q,U},ntime] tells us how to turn the
		# barrel signals into detector outputs
		sig_det_dc    = np.einsum("dbct,cbt->dt", resp[:,0], sig_barrel_dc)
		sig_det_delay = np.einsum("dbct,cbt->dt", resp[:,0], sig_barrel_delay)
		sig_tot = sig_det_dc + sig_det_delay
		return sig_tot

ofile = args.ofile
order = args.interpol
with bench.show("read"):
	# Read our sky cube, which should be [{T,Q,U},ndelay,y,x]
	sky, spec_wcs = pixutils.read_map(args.sky_map)
with bench.show("prefilter"):
	# Prefilter to make interpolation faster
	if order > 1: sky = utils.interpol_prefilter(sky, order)

sgen = ScanGenerator()
with bench.show("orbit"):
	offset = int(sgen.scan_period/sgen.sample_period * args.scan)
	num    = int(sgen.scan_period/sgen.sample_period * args.nscan)
	oparam = sgen.gen_orbit(i0=offset, n=num, step=args.step)
with bench.show("point"):
	point  = sgen.gen_pointing(oparam)
with bench.show("resp"):
	resp   = sgen.gen_det_resp(point)
with bench.show("pixels"):
	pix    = sgen.gen_pixels(point, sky.wcs, spec_wcs)
with bench.show("signal"):
	signal = sgen.gen_signal(sky, pix, resp, order=np.abs(order))
with bench.show("write"):
	with h5py.File(ofile,"w") as hfile:
		hfile["point"] = point.point
		hfile["pos"]   = point.pos
		hfile["gamma"] = point.gamma
		hfile["tod"]   = signal
		hfile["pix"]   = pix
