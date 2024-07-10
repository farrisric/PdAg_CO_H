from acat.adsorption_sites import ClusterAdsorptionSites
from acat.adsorbate_coverage import ClusterAdsorbateCoverage
from acat.build import add_adsorbate_to_site
from ase.cluster import Octahedron
from ase.visualize import view
from acat.ga.adsorbate_operators import (AddAdsorbate, RemoveAdsorbate,
                                         MoveAdsorbate, ReplaceAdsorbate,
                                         SimpleCutSpliceCrossoverWithAdsorbates)
from acat.ga.particle_mutations import (RandomPermutation, COM2surfPermutation,
                                        Rich2poorPermutation, Poor2richPermutation)
from acat.ga.particle_mutations import RandomPermutation
from ase.ga.offspring_creator import OperationSelector
import numpy as np
from acat.build.adlayer import min_dist_coverage_pattern
from ase import Atoms
from ase.optimize import BFGS
from ase.io.trajectory import TrajectoryWriter
from ocpmodels.common.relaxation.ase_utils import OCPCalculator
from ase.io import read
from ase.constraints import FixAtoms
from ase.calculators.emt import EMT
import copy
import sys
import numpy as np
import random
from ase.units import kB as boltzmann_constant
from mace.calculators import mace_mp

n_atoms = int(sys.argv[1])
temperature = 50
calculator = mace_mp(model="/work/g15farris/2023-12-03-mace-128-L1_epoch-199.model")

def _acceptance_condition(potential_diff: float) -> bool:
    if potential_diff <= 0:
        return True
    elif temperature <= 1e-16:
        return False
    else:
        p = np.exp(-potential_diff / (boltzmann_constant * temperature))
        return p > _next_random_number()
    
def _next_random_number() -> float:
    """ Returns the next random number from the PRNG. """
    return random.random()

def trial_step(atoms, op_selector, calc, traj_out):
    lowest_energy = atoms.get_potential_energy()
    new_atoms = atoms
    new_atoms.info['data'] = {'tag': None}
    new_atoms.info['confid'] = 1
    # prob, operations = list_operations
    # n_ops = np.random.geometric(0.35, size=1).item()+len(operations)
    
    operation = op_selector.get_operator()
    new_atoms, _ = operation.get_new_individual([new_atoms])
    #new_atoms.info['data'] = {'tag': None}
    #new_atoms.info['confid'] = 1        
    #new_atoms.set_constraint(fix_indices)
    new_atoms = relax(new_atoms, calc)
     
    potential_diff = new_atoms.info['key_value_pairs']['raw_score'] - atoms.info['key_value_pairs']['raw_score']
    if _acceptance_condition(potential_diff):
        lowest_energy = new_atoms.info['key_value_pairs']['raw_score']
        atoms = new_atoms
        atoms.info['data'] = {'tag': None}
        atoms.info['confid'] = 1
        traj_out.write(atoms)
    return atoms, lowest_energy

def run(atoms, steps, operator, calc):
    traj_out = TrajectoryWriter(f'opt_run_AgPdH_{n_atoms}.traj')
    
    for i in range(steps):
        atoms, energy_step = trial_step(atoms, operator, calc, traj_out)

        if i % 10 == 0:
            print(f'Step : {i}, Energy : {energy_step}')

no_improvements = 0
#calculator = EMT() #OCPCalculator(checkpoint_path="/work/g15farris/ocp_pot/painn_h512_s2ef_all.pt")

atoms = Octahedron('Ag', 9, 3)
for x in np.random.choice(len(atoms), n_atoms, replace=False):
    atoms[x].symbol = 'Pd'
        
atoms = Atoms(positions=atoms.positions, symbols=atoms.symbols)
atoms.center(5)
atoms.info['data'] = {'tag': None}
atoms.info['confid'] = 1
atoms, _ = AddAdsorbate('H', num_muts=n_atoms*3).get_new_individual([atoms])

atoms.info['data'] = {'tag': None}
atoms.info['confid'] = 1

species = ['H']
sas = ClusterAdsorptionSites(atoms, composition_effect=False)
soclist = ([5,5,5,3,2,1],
        [Rich2poorPermutation(elements=['Ag', 'Pd'], num_muts=1),
        Poor2richPermutation(elements=['Ag', 'Pd'], num_muts=1),
        RandomPermutation(elements=['Ag', 'Pd'], num_muts=1),
        MoveAdsorbate(species, adsorption_sites=sas, num_muts=1),
        MoveAdsorbate(species, adsorption_sites=sas, num_muts=2),
        MoveAdsorbate(species, adsorption_sites=sas, num_muts=3),
        #SimpleCutSpliceCrossoverWithAdsorbates(species, keep_composition=True,
        #                                       adsorption_sites=sas),
        ])
op_selector = OperationSelector(*soclist)

chem_pots = {'O2' : -0.1}

atoms.calc = calculator
opt = BFGS(atoms, logfile=None)
opt.run(fmax=0.03)
# references = {'O2' : 5.940566, 'NP' : atoms.get_potential_energy()}
    
def relax(atoms, calc, fmax=0.1):
    atoms.info['key_value_pairs'] = {}
    atoms.calc = calc
    opt = BFGS(atoms, logfile=None)
    opt.run(fmax=fmax)
    
    Epot = atoms.get_potential_energy()
    # num_H = len([s for s in atoms.symbols if s == 'O'])
    # mutot = num_H * (references['O2']/2 + chem_pots['O2']/2)
    f = Epot #- references['NP'] - mutot
    
    atoms.info['key_value_pairs']['raw_score'] = f
    atoms.info['key_value_pairs']['potential_energy'] = Epot
    
    return atoms

relax(atoms, calculator)
#fix_indices = FixAtoms(indices=[atom.index for atom in atoms if atom.symbol in ['Ag', 'Zn']])
#atoms.set_constraint(fix_indices)
run(atoms, 20000, op_selector, calculator)
