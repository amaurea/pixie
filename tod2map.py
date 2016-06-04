import numpy as np, argparse, pixie, h5py
from enlib import enmap, utils
parser = argparse.ArgumentParser()
parser.add_argument("tod")
parser.add_argument("odir")
args = parser.parse_args()

nfreq  = 480
nblock =   4
ndelay = nfreq*nblock
nspin  = 8

tod = pixie.read_tod(args.tod)
d   = tod.signal[0,:-1]
d   = d.reshape(-1,nspin,ndelay)

hfile = h5py.File("test.hdf","w")
hfile["d"] = d

# FIXME: Should ensure periodicity at this point

## Adjust phase to compensate for sky motion during a spin
d  = pixie.froll(d, np.arange(nspin) [None,:,None]/float(nspin), 0)
## Adjust phase to compensate for spin during strokes
d  = pixie.froll(d, np.arange(ndelay)[None,None,:]/float(ndelay),1)
#d = pixie.froll(d.reshape(-1,nspin*ndelay), np.arange(nspin*ndelay)/float(nspin*ndelay), 0).reshape(-1,nspin,ndelay)

# FIXME: We can't decompose it like this. Look at it as an interpolation
# problem in a 3d box. Can still use fourier interpolation due to periodicity
# + equi-spaced steps, I think.

hfile["d2"] = d

# Ok, at this point the dimensions are independent. The FT of the
# delay dimension is a projection of the spectrum. The spin-2 components
# of the spin direction describes Q and U. But what about T? Double
# barrel operation is normally not sensitive to T (except the DC component).
# The DC component just becomes an overall offset of the spectrum, which we
# get rid of by demanding spec[0]=0 or spec[-1]=0

# Split into left-going and right-going scan and positive and negative frequencies
d = d.reshape(-1,nspin,nblock,nfreq)

hfile["d3"] = d
hfile.close()

