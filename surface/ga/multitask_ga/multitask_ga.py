from acat.settings import adsorbate_elements
from acat.adsorption_sites import SlabAdsorptionSites
from acat.adsorbate_coverage import SlabAdsorbateCoverage
from acat.build.ordering import RandomOrderingGenerator as ROG
from acat.build.adlayer import OrderedPatternGenerator as OPG
from acat.ga.adsorbate_operators import (ReplaceAdsorbateSpecies,
                                         CatalystAdsorbateCrossover)
from acat.ga.slab_operators import (CutSpliceSlabCrossover,
                                    RandomSlabPermutation,
                                    RandomCompositionMutation)
from acat.ga.group_operators import (AdsorbateGroupSubstitute,
                                     AdsorbateGroupPermutation)
from acat.ga.multitasking import (MultitaskPopulation,
                                  MultitaskRepetitionConvergence)
from ase.ga.standard_comparators import SequentialComparator, StringComparator
from ase.ga.offspring_creator import OperationSelector
from ase.ga.population import Population, RankFitnessPopulation
from ase.ga.convergence import GenerationRepetitionConvergence
from ase.ga.utilities import closest_distances_generator, get_nnmat
from ase.ga.data import DataConnection, PrepareDB
from ase.io import read, write
from ase.build import fcc111
from ase.calculators.emt import EMT
from ase.optimize import BFGS
from collections import defaultdict
from multiprocessing import Pool
from random import randint
import numpy as np
import time
import os
from mace.calculators import mace_mp

# Define population
# Recommend to choose a number that is a multiple of the number of cpu
pop_size = 50
macemp = mace_mp()


# Define the tasks. In this case we use 10 different chemical potentials of CH4
tasks = np.arange(-6., 0., 0.5)

# Generate 50 Ni48Pt16 slabs with random orderings
slab = fcc111('Ag', (4, 4, 4), vacuum=5., orthogonal=True, periodic=True)
slab_ids = list(range(len(slab)))
rog = ROG(slab, elements=['Ag', 'Pd'],
          composition={'Ag': 0.75, 'Pd': 0.25},
          trajectory='starting_generation.traj')
rog.run(num_gen=pop_size)

# Get the adsorption sites. Composition does not affect GA operations
sas = SlabAdsorptionSites(slab, surface='fcc111', ignore_sites='bridge',
                          composition_effect=False)

# Generate random coverage on each slab and save the groupings
species = ['H']
images = read('starting_generation.traj', index=':')
opg = OPG(images, adsorbate_species=species, surface='fcc111',
          adsorption_sites=sas, max_species=2, allow_odd=True,
          remove_site_shells=1, save_groups=True,
          trajectory='patterns.traj', append_trajectory=True)
opg.run(max_gen=pop_size, unique=True)
patterns = read('patterns.traj', index=':')

# Instantiate the db
db_name = 'ridge_Ag48Pd16_ads_multitask.db'

db = PrepareDB(db_name, cell=slab.cell, population_size=pop_size)

for atoms in patterns:
    if 'data' not in atoms.info:
        atoms.info['data'] = {'tag': None}
    db.add_unrelaxed_candidate(atoms, data=atoms.info['data'])

# Connect to the db
db = DataConnection(db_name)

# Define operators
soclist = ([3, 3, 3, 3, 3],
           [RandomSlabPermutation(allowed_indices=slab_ids),
            RandomCompositionMutation(allowed_indices=slab_ids),
            AdsorbateGroupSubstitute(species, max_species=2,
                                     adsorption_sites=sas,
                                     remove_site_shells=1),
            AdsorbateGroupPermutation(species, adsorption_sites=sas,
                                      remove_site_shells=1),
            CatalystAdsorbateCrossover(),])
op_selector = OperationSelector(*soclist)

# Define comparators
comp = StringComparator('potential_energy')

def get_ads(atoms):
    """Returns a list of adsorbate names and corresponding indices."""

    if 'data' not in atoms.info:
        atoms.info['data'] = {'tag': None}
    if 'adsorbates' in atoms.info['data']:
        adsorbates = atoms.info['data']['adsorbates']
    else:
        cac = SlabAdsorbateCoverage(atoms, adsorption_sites=sas)
        adsorbates = [t[0] for t in cac.get_adsorbates()]

    return adsorbates

def vf(atoms):
    """Returns the descriptor that distinguishes candidates in the
    niched population."""

    return len(get_ads(atoms))

# Give fittest candidates at different coverages equal fitness.
# Use this to find global minimum at each adsorbate coverage
pop = MultitaskPopulation(data_connection=db,
                          population_size=pop_size,
                          num_tasks=len(tasks),
                          comparator=comp,
                          exp_function=True,
                          logfile='log_multitask.txt')

# Normal fitness ranking irrespective of adsorbate coverage
#pop = Population(data_connection=db,
#                 population_size=pop_size,
#                 comparator=comp,
#                 logfile='log.txt')

# Set convergence criteria
cc = GenerationRepetitionConvergence(pop, 5)

# Calculate chemical potentials
chem_pots = {'H2': tasks}

# Define the relax function
def relax(atoms, single_point=False):
    atoms.calc = macemp
    if not single_point:
        opt = BFGS(atoms, logfile=None)
        opt.run(fmax=0.05)

    Epot = atoms.get_potential_energy()
    num_H = len([s for s in atoms.symbols if s == 'H'])
    mutot = chem_pots['H2'] / 2
    f = -(Epot - mutot)

    atoms.info['key_value_pairs']['raw_score'] = f
    atoms.info['key_value_pairs']['potential_energy'] = Epot

    return atoms

# Relax starting generation
def relax_an_unrelaxed_candidate(atoms):
    if 'data' not in atoms.info:
        atoms.info['data'] = {'tag': None}
    nncomp = atoms.get_chemical_formula(mode='hill')
    print('Relaxing ' + nncomp)

    return relax(atoms, single_point=False) # Single point only for testing

# Create a multiprocessing Pool
pool = Pool(25)#os.cpu_count())
# Perform relaxations in parallel.
relaxed_candidates = pool.map(relax_an_unrelaxed_candidate,
                              db.get_all_unrelaxed_candidates())
pool.close()
pool.join()
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
            # Pure or bare slabs are not considered
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
    pool = Pool(25)#os.cpu_count())
    # Perform procreations in parallel. Especially useful when
    # using adsorbate operators which requires site identification
    relaxed_candidates = pool.map(procreation, range(pop_size))
    pool.close()
    pool.join()
    db.add_more_relaxed_candidates(relaxed_candidates)

    # Update the population to allow new candidates to enter
    pop.update()