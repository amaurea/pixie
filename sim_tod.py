import numpy as np, argparse, pixie
from enlib import mpi, utils, log, fft
parser = argparse.ArgumentParser()
parser.add_argument("orbits")
parser.add_argument("odir")
parser.add_argument("-C", "--config", type=str, default=None)
parser.add_argument("-s", "--seed",   type=int, default=0)
args = parser.parse_args()

fft.engine = "fftw"
orbits = pixie.parse_ints(args.orbits)
comm   = mpi.COMM_WORLD
L      = log.init(rank=comm.rank, shared=False, level='DEBUG')

# Build our actual simulator. This is shared between tods
config = pixie.load_config(args.config)
sim    = pixie.PixieSim(config)
utils.mkdir(args.odir)

for ind, orbit in enumerate(orbits):
	orbit = orbits[ind]
	L.info("orbit %3d" % orbit)
	np.random.seed([args.seed, orbit])
	tod   = sim.sim_tod(orbit, comm=comm)
	if comm.rank == 0:
		pixie.write_tod(args.odir + "/tod%03d.hdf" % orbit, tod)
	del tod
