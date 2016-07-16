import numpy as np, argparse, pixie
from enlib import mpi, utils, log, fft
parser = argparse.ArgumentParser()
parser.add_argument("orbits")
parser.add_argument("odir")
parser.add_argument("-C", "--config", type=str, default=None)
args = parser.parse_args()

fft.engine = "fftw"
orbits = pixie.parse_ints(args.orbits)
comm   = mpi.COMM_WORLD
L      = log.init(rank=comm.rank, shared=False, level='DEBUG')

# Build our actual simulator. This is shared between tods
config = pixie.load_config(args.config)
sim    = pixie.PixieSim(config)
utils.mkdir(args.odir)

for ind in range(comm.rank, len(orbits), comm.size):
	orbit = orbits[ind]
	L.info("orbit %3d" % orbit)
	tod   = sim.sim_tod(orbit)
	pixie.write_tod(args.odir + "/tod%03d.hdf" % orbit, tod)
	del tod
