from mace.calculators import mace_mp
from ase import build
from ase.md import Langevin
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase import units
from ase.cluster import Octahedron
import time
import torch

macemp = mace_mp(model="/work/g15farris/2023-12-03-mace-128-L1_epoch-199.model", device=torch.cuda.current_device(),default_dtype='float64')

atoms = build.molecule('H2O')
atoms = Octahedron('Pd', 9,1)
atoms.calc = macemp

# Initialize velocities.
T_init = 300  # Initial temperature in K
MaxwellBoltzmannDistribution(atoms, T_init * units.kB)

t0 = time.time()
# Set up the Langevin dynamics engine for NVT ensemble.
dyn = Langevin(atoms, 0.5 * units.fs, T_init * units.kB, 0.001)
n_steps = 200 # Number of steps to run
dyn.run(n_steps)

t1= time.time()
print(t1-t0)
