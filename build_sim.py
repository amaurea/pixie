import numpy as np, argparse
from enlib import enmap, curvedsky, powspec, bunch, coordinates, utils, bench
parser = argparse.ArgumentParser()
#parser.add_argument("template")
#parser.add_argument("powspec")
#parser.add_argument("omap")
parser.add_argument("-T", type=float, default=2.7260)
parser.add_argument("--dfreq", type=float, default=15)
parser.add_argument("--nbin",  type=float, default=400)
parser.add_argument("--mirror",type=float, default=0.5)
args = parser.parse_args()

#template = enmap.read_map(args.template)
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
		#self.scan_period   = 16*self.spin_period
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
		"""Use orbital positions into pointing and orientation on the sky."""
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
		np.savetxt("bar.txt", zvec[:,0].T)
		point = utils.rect2ang(zvec,axis=0)
		# Get the polarization orientation on the sky
		gamma = np.arctan2(xvec[2], -zvec[1]*xvec[0]+zvec[0]*xvec[1])
		# And the delay at each time
		delay = self.delay_amp * np.sin(oparam.delay)
		return bunch.Bunch(point=point, gamma=gamma, delay=delay)
	def gen_det_resp(self, point):
		"""Compute the on-sky T,Q,U sensitivity of each detector for each pointing.
		Will be [t,ndet,{0,delay},nbarrel,{T,Q,U}] because every detector gets contributions from
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
	#def gen_optics(self):
	#	"""Compute the effect of the telescope polarization filters and mirrors.
	#	Follows vertial and horizontal polarization from each barrel through the
	#	optics until the detectors. Returns the mapping from incoming radiation
	#	in each barrel to incoming radiation in each feedhorn separately for
	#	the radiation on each side of the moving mirror. The total transfer
	#	matrix will be a linear combination of these based on the stroke.
	#	
	#	Returns [mirror side,{barrel A, barrel B}*{hor,vert}]
	#	"""
	#	# Basis is [Ah,Av,Bh,Bv], where A and B are the two horns, and h and v
	#	# are the two polarization directions.
	#	# Transmits vertial, so vertial moves to the other horn
	#	# could be more compactly, but less readably, written as roll(eye(4)[::-1],1,0)
	#	Mf = np.array([
	#		[1,0,0,0],
	#		[0,0,0,1],
	#		[0,0,1,0],
	#		[0,1,0,0]])
	#	# Rotation from + to x basis
	#	Mr = 2**-0.5*np.array([
	#		[ 1, 1, 0, 0],
	#		[-1, 1, 0, 0],
	#		[ 0, 0, 1, 1],
	#		[ 0, 0,-1, 1]])
	#	Mpre = Mf.dot(Mr.dot(Mf))
	#	res = np.einsum("ij,aj,jk->aik", Mpre.T, [[1,1,0,0],[0,0,1,1]], Mpre)
	#	return res

sgen = ScanGenerator()
with bench.show("orbit"):
	oparam = sgen.gen_orbit(step=10000, n=10000)
with bench.show("point"):
	point  = sgen.gen_pointing(oparam)
with bench.show("resp"):
	resp   = sgen.gen_det_resp(point)

np.savetxt("foo.txt", point.point[0].T)

#
#
#
#def deproject(a, b):
#	a = np.asarray(a)
#	return a - a*np.sum(a*b,axis=-1)[...,None]
#
#def generate_pointing(scan_params, t0, nsamp):
#	t = t0 + np.arange(nsamp)*scan_params.sample_period
#	toff = t - scan_params.ref_ctime
#	ang_orbit = scan_params.orbit_phase   + 2*np.pi*toff/scan_params.orbit_period
#	ang_scan  = scan_params.scan_phase    + 2*np.pi*toff/scan_params.scan_period
#	ang_spin  = scan_params.spin_phase    + 2*np.pi*toff/scan_params.spin_period
#	ang_delay = scan_params.delay_phase   + 2*np.pi*toff/scan_params.delay_period
#	delay = scan_params.delay_amp * np.sin(ang_delay)
#
#	# Our pointing is given by a series of rotations:
#	# 1. Rotate the horn away from the spin axis
#	#    Rzyz(horn_phi,horn_theta,ang_spin)
#	# 2. Rotate the spin axis to its position in the scan
#	#    Rzyz(ang_scan,opening_angle,0)
#	# 3. Rotate the scan axis to its position along the orbit
#	#    Rzyz(ang_orbit, 0, 0)
#	# The #3 can be combined with #2.
#	R_horn = coordinates.euler_mat(scan_params.horn_offs.T)
#	R_rest = coordinates.euler_mat([
#		ang_scan,
#		np.repeat(scan_params.opening_angle,nsamp),
#		ang_orbit])
#	vecs = np.array([[0,0,1],[1,0,0]])
#	pointing = np.einsum("tij,hvj->htvi", R_rest,
#			np.einsum("hij,vj->hvi", R_horn, vecs))
#	# The result is [horn,time,vec,{x,y,z}].
#	# Construct tangent coordinate system, and recover polarization angle.
#	# This could be more elegant.
#	z_vec, oldx_vec = np.rollaxis(pointing,2)
#	x_vec = deproject([0,0,1], z_vec)
#	y_vec = -np.cross(x_vec, z_vec)
#	psi_ang = np.arctan2(np.sum(oldx_vec*y_vec,-1), np.sum(oldx_vec*x_vec,-1))
#	# Convert the boresight into spherical coordiantes. This
#	# represents this barrel's pointing. We specify zenith=False
#	# to get angles counting up from the equator. These will
#	# correpond to equatorial heliocentric [horn,time,{b,l}]
#	print z_vec
#	z_ang = utils.rect2ang(z_vec.T, zenith=False).T
#
#	return bunch.Bunch(toff=toff, point=z_ang, psi=psi_ang, delay=delay)
#
#pointing = generate_pointing(scan_params, scan_params.ref_ctime, 10000)
#print pointing
