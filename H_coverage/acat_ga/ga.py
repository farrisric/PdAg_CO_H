from acat.settings import adsorbate_elements
from acat.adsorption_sites import ClusterAdsorptionSites
from acat.adsorbate_coverage import ClusterAdsorbateCoverage
from acat.build.ordering import RandomOrderingGenerator as ROG
from acat.build.adlayer import min_dist_coverage_pattern
from acat.ga.adsorbate_operators import (AddAdsorbate, RemoveAdsorbate,
                                         MoveAdsorbate, ReplaceAdsorbate,
                                         SimpleCutSpliceCrossoverWithAdsorbates)
# Import particle_mutations from acat instead of ase to get the indexing-preserved version
from acat.ga.particle_mutations import (RandomPermutation, COM2surfPermutation,
                                        Rich2poorPermutation, Poor2richPermutation)
from ase.ga.particle_comparator import NNMatComparator
from ase.ga.standard_comparators import SequentialComparator, StringComparator
from ase.ga.offspring_creator import OperationSelector
from ase.ga.population import Population, RankFitnessPopulation
from ase.ga.convergence import GenerationRepetitionConvergence
from ase.ga.utilities import closest_distances_generator, get_nnmat
from ase.ga.data import DataConnection, PrepareDB
from ase.io import read, write
from ase.cluster import Octahedron
from ase.calculators.emt import EMT
from ase.optimize import BFGS
from collections import defaultdict
from random import uniform, randint
import numpy as np
import time
import os
from mace.calculators import mace_mp

macemp =  mace_mp(model="/work/g15farris/2023-12-03-mace-128-L1_epoch-199.model")
# Define population
# Recommend to choose a number that is a multiple of the number of cpu
pop_size = 50

# Generate 50 icosahedral Ni110Pt37 nanoparticles with random orderings
particle = Octahedron('Pd', 9 ,3)
particle.center(vacuum=5.)
rog = ROG(particle, elements=['Pd', 'Ag'],
          composition={'Ag': 300, 'Pd': 105},
          trajectory='starting_generation.traj')
rog.run(num_gen=pop_size)

# Generate random coverage 
species = ['H']
patterns = []
for images in read('starting_generation.traj', ':'):
    dmin = uniform(3.5, 8.5)
    pattern = min_dist_coverage_pattern(images, adsorbate_species=species,
                                        min_adsorbate_distance=dmin)
    patterns.append(pattern)
    
# Get the adsorption sites. Composition does not affect GA operations
sas = ClusterAdsorptionSites(particle, composition_effect=False)

# Instantiate the db
db_name = 'PdAg_H.db'

db = PrepareDB(db_name, cell=particle.cell, population_size=pop_size)

for atoms in patterns:
    if 'data' not in atoms.info:
        atoms.info['data'] = {'tag': None}
    db.add_unrelaxed_candidate(atoms, data=atoms.info['data'])
    
# Connect to the db
db = DataConnection(db_name)

# Define operators
soclist = ([1,1,2,1, 1, 1, 2],
           [
            Rich2poorPermutation(elements=['Ag', 'Pd'], num_muts=5),
            Poor2richPermutation(elements=['Ag', 'Pd'], num_muts=5),
            RandomPermutation(elements=['Ag', 'Pd'], num_muts=5),
            AddAdsorbate(species, adsorption_sites=sas, num_muts=5),
            RemoveAdsorbate(species, adsorption_sites=sas, num_muts=5),
            MoveAdsorbate(species, adsorption_sites=sas, num_muts=5),
            #ReplaceAdsorbate(species, adsorption_sites=sas, num_muts=5),
            SimpleCutSpliceCrossoverWithAdsorbates(species, keep_composition=True,
                                                   adsorption_sites=sas),])
op_selector = OperationSelector(*soclist)

# Define comparators
comp = SequentialComparator([StringComparator('potential_energy'),
                             NNMatComparator(0.2, ['Pd', 'Ag'])],
                            [0.5, 0.5])

def get_ads(atoms):
    """Returns a list of adsorbate names and corresponding indices."""

    if 'data' not in atoms.info:
        atoms.info['data'] = {'tag': None}
    if 'adsorbates' in atoms.info['data']:
        adsorbates = atoms.info['data']['adsorbates']
    else:
        cac = ClusterAdsorbateCoverage(atoms)
        adsorbates = [t[0] for t in cac.get_adsorbates()]

    return adsorbates

def vf(atoms):
    """Returns the descriptor that distinguishes candidates in the
    niched population."""

    return len(get_ads(atoms))

pop = RankFitnessPopulation(data_connection=db,
                            population_size=pop_size,
                            comparator=comp,
                            variable_function=vf,
                            exp_function=True,
                            logfile='log.txt')

# pop = Population(data_connection=db,
#                 population_size=pop_size,
#                 comparator=comp,
#                 logfile='log.txt')

cc = GenerationRepetitionConvergence(pop, 5)

chem_pots = {'H2': -0.5}
references = {'H2': -6.5244903564453125}


# Define the relax function
def relax(atoms, single_point=True):
    atoms.center(vacuum=5.)
    atoms.calc = macemp
    if not single_point:
        opt = BFGS(atoms, logfile=None)
        opt.run(fmax=0.1)
        
    Epot = atoms.get_potential_energy()
    num_H = len([s for s in atoms.symbols if s == 'H'])
    mutot = num_H * (references['H2']/2 + chem_pots['H2']/2)
    f = Epot - mutot
    
    atoms.info['key_value_pairs']['raw_score'] = f
    atoms.info['key_value_pairs']['potential_energy'] = Epot

    # Parallelize nnmat calculations to accelerate NNMatComparator
    atoms.info['data']['nnmat'] = get_nnmat(atoms)

    return atoms

# Relax starting generation
def relax_an_unrelaxed_candidate(atoms):
    if 'data' not in atoms.info:
        atoms.info['data'] = {'tag': None}
    nncomp = atoms.get_chemical_formula(mode='hill')
    print('Relaxing ' + nncomp)

    return relax(atoms, single_point=True) # Single point only for testing

# Create a multiprocessing Pool
# Perform relaxations in parallel. Especially
# useful when running GA on large nanoparticles
relaxed_candidates = list(map(relax_an_unrelaxed_candidate,
                              db.get_all_unrelaxed_candidates()))

db.add_more_relaxed_candidates(relaxed_candidates)
pop.update()

# Number of generations
num_gens = 1000

# Below is the iterative part of the algorithm
gen_num = db.get_generation_number()
for i in range(num_gens):
    # Check if converged
    if cc.converged():
        print('Converged')
        break
    print('Creating and evaluating generation {0}'.format(gen_num + i))

    def procreation(x):
        # Select an operator and use it
        op = op_selector.get_operator()
        while True:
            # Assign rng with a random seed
            np.random.seed(randint(1, 10000))
            pop.rng = np.random
            # Select parents for a new candidate
            p1, p2 = pop.get_two_candidates()
            parents = [p1, p2]
            # Pure or bare nanoparticles are not considered
            if len(set(p1.numbers)) < 3:
                continue
            offspring, desc = op.get_new_individual(parents)
            # An operator could return None if an offspring cannot be formed
            # by the chosen parents
            if offspring is not None:
                break
        nncomp = offspring.get_chemical_formula(mode='hill')
        print('Relaxing ' + nncomp)
        if 'data' not in offspring.info:
            offspring.info['data'] = {'tag': None}

        return relax(offspring, single_point=True) # Single point only for testing

    # Create a multiprocessing Pool
    # Perform procreations in parallel. Especially useful when
    # using adsorbate operators which requires site identification
    relaxed_candidates = list(map(procreation, range(pop_size)))
    db.add_more_relaxed_candidates(relaxed_candidates)

    # Update the population to allow new candidates to enter
    pop.update()
