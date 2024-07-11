from mace.calculators import mace_mp
from ase.io import read
from ase.build import molecule
from ase.optimize import BFGS

macemp =  mace_mp(model="/work/g15farris/2023-12-03-mace-128-L1_epoch-199.model")

def get_references():
    H2 = molecule('H2')
    H2.calc = macemp
    BFGS(H2).run(fmax=0.03)
    H2 = H2.get_potential_energy()

    atoms = read('/home/g15farris/AdsGO/PdAg_CO_H/bare_nps/xyz/Ag300Pd105_LEH.xyz')
    atoms.calc = macemp
    BFGS(atoms).run(fmax=0.03)
    NP = atoms.get_potential_energy()
    
    return {'NP':NP, 'H2':H2}

