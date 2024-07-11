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
from ase.io import read
from ase.constraints import FixAtoms
from ase.calculators.emt import EMT
import copy
import sys
import numpy as np
import random
from ase.units import kB as boltzmann_constant
from mace.calculators import mace_mp
import torch
from references import get_references

cuda = torch.cuda.current_device()
temperature = 50
calculator =  mace_mp(model="/work/g15farris/2023-12-03-mace-128-L1_epoch-199.model", device=cuda)
references = get_references()
chem_pots = {'H2' : -1}

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

def relax(atoms, calc, fmax=0.1):
    atoms.info['key_value_pairs'] = {}
    atoms.calc = calc
    opt = BFGS(atoms, logfile=None)
    opt.run(fmax=fmax)
    
    Epot = atoms.get_potential_energy()
    num_H = len([s for s in atoms.symbols if s == 'H'])
    mutot = num_H * (references['H2']/2 + chem_pots['H2']/2)
    f = Epot - references['NP'] - mutot
    
    atoms.info['key_value_pairs']['raw_score'] = f
    atoms.info['key_value_pairs']['potential_energy'] = Epot
    
    return atoms

def trial_step(atoms, op_selector, calc, traj_out):
    lowest_energy = atoms.get_potential_energy()
    new_atoms = atoms
    new_atoms.info['data'] = {'tag': None}
    new_atoms.info['confid'] = 1
    operation = op_selector.get_operator()
    new_atoms, _ = operation.get_new_individual([new_atoms])
    new_atoms = relax(new_atoms, calc)
    potential_diff = new_atoms.info['key_value_pairs']['raw_score'] - atoms.info['key_value_pairs']['raw_score']
    
    if _acceptance_condition(potential_diff):
        lowest_energy = new_atoms.info['key_value_pairs']['raw_score']
        atoms = new_atoms
        atoms.info['data'] = {'tag': None}
        atoms.info['confid'] = 1
        traj_out.write(atoms)
    return atoms, lowest_energy

def run(atoms, steps, operator, calc, traj_out):
    for i in range(steps):
        atoms, energy_step = trial_step(atoms, operator, calc, traj_out)

        if i % 10 == 0:
            print(f'Step : {i}, Energy : {energy_step}')

no_improvements = 0
atoms = read('/home/g15farris/AdsGO/PdAg_CO_H/bare_nps/xyz/Ag300Pd105_LEH.xyz')
        
atoms = Atoms(positions=atoms.positions, symbols=atoms.symbols)
atoms.center(5)
relax(atoms, calculator)
atoms.info['data'] = {'tag': None}
atoms.info['confid'] = 1
atoms = min_dist_coverage_pattern(atoms, adsorbate_species=['H'],
                                  min_adsorbate_distance=3.5)

atoms.info['data'] = {'tag': None}
atoms.info['confid'] = 1

species = ['H']
sas = ClusterAdsorptionSites(atoms, composition_effect=False)
soclist = ([1,1,1,1,1,1],
        [Rich2poorPermutation(elements=['Ag', 'Pd'], num_muts=1),
        Poor2richPermutation(elements=['Ag', 'Pd'], num_muts=1),
        RandomPermutation(elements=['Ag', 'Pd'], num_muts=1),
        MoveAdsorbate(species, adsorption_sites=sas, num_muts=1),
        AddAdsorbate(species, adsorption_sites=sas, num_muts=1),
        RemoveAdsorbate(species, adsorption_sites=sas, num_muts=1)
        ])
op_selector = OperationSelector(*soclist)

atoms.calc = calculator
opt = BFGS(atoms, logfile=None)
opt.run(fmax=0.03)

relax(atoms, calculator)
#fix_indices = FixAtoms(indices=[atom.index for atom in atoms if atom.symbol in ['Ag', 'Zn']])
#atoms.set_constraint(fix_indices)
traj_out = TrajectoryWriter(f'opt_run_Ag305Pd100_H_MACEGPU1.traj')
run(atoms, 20000, op_selector, calculator, traj_out)
