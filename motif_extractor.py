import os, re, time, math
import pickle
import numpy as np
import pandas as pd

from collections import Counter
from multiprocessing import Pool

from pymatgen.core.structure import Structure
from pymatgen.core.composition import Composition

from pymatgen.analysis.local_env import CrystalNN
from pymatgen.analysis.chemenv.coordination_environments.chemenv_strategies import SimplestChemenvStrategy, MultiWeightsChemenvStrategy
from pymatgen.analysis.chemenv.coordination_environments.structure_environments import LightStructureEnvironments 
from pymatgen.analysis.chemenv.coordination_environments.coordination_geometry_finder import LocalGeometryFinder

from robocrys.condense.fingerprint import get_site_fingerprints, get_structure_fingerprint

from motif_type_info import most_occ_envs, motif_type_nmbrs

import logging 

logging.basicConfig(filename='error.log', level=logging.ERROR)

lgf = LocalGeometryFinder()
strategy = SimplestChemenvStrategy(distance_cutoff=1.4, angle_cutoff=0.3 , additional_condition =3)


class MotifFeaturesDetails:
    def __init__(self, structure_file):
        self.filename = os.path.basename(structure_file).replace('.cif', '')
        self.structure = Structure.from_file(structure_file)
        self.elements = [self.structure.sites[i].species_string for i in range(self.structure.num_sites)]
        self.oxi_state = self.structure.composition.oxi_state_guesses()
        self.anions = [k for k, v in self.oxi_state[0].items() if v < 0] if self.oxi_state else ['F', 'O', 'N', 'Cl', 'Br', 'I', 'S']
        if not (set(self.anions) & set(self.elements)):
            self.anions = []
        self.final_el_list = [el for el in self.elements if el not in self.anions]
        lgf.setup_structure(structure=self.structure)
        self.se = lgf.compute_structure_environments(maximum_distance_factor=1.41, excluded_atoms=self.anions)
        self.lse = LightStructureEnvironments.from_structure_environments(strategy, self.se)
    
    def get_motif_details(self):
        motif_type_list, coordination_env_list = [], []
        for i in range(self.structure.num_sites):
            y = self.lse.coordination_environments[i]
            coordination_env_list.append(y)
        for i in range(len(coordination_env_list)):
            if coordination_env_list[i] == None:
                continue
            else:
                # filter the motifs bases on continuous symmetry measure
                result = min(coordination_env_list[i], key=lambda x: x['csm'])
                # collect only those symbols that matches the keys in most_occuring_env
                if result['ce_symbol'] in most_occ_envs.keys():
                    motif_type = most_occ_envs[result['ce_symbol']]
                    motif_type_list.append(motif_type)
        return coordination_env_list, motif_type_list
    
    def get_motif_comp(self):
        central_atom, neighbor_finding = [], []
        coordination_env_list, motif_type_list = self.get_motif_details()
        for i in range(len(motif_type_list)):
            site = self.structure[i]
            central_atom.append(site.species_string)
            #print(site)
            neighbors = strategy.get_site_neighbors(site)
            #print(neighbors)
            neighboring_atoms = [i.species_string for i in neighbors]
            neighbor_finding.append(neighboring_atoms)

        neighbor_finding = [sorted(i) for i in neighbor_finding]
        final_motif_list = [str(i) + ''.join('%s%d' % t for t in Counter(j).items()) for i, j in 
                            zip(central_atom, neighbor_finding)]
        return final_motif_list, motif_type_list
    
    def vectorize_motif(motif, motif_type_nmbrs):
        vector = np.zeros(len(motif_type_nmbrs) + 1)
        vector[motif_type_nmbrs.get(motif, 0) - 1] = 1
        return vector
    
    def get_structure_fingerprint(self):
        return get_structure_fingerprint(self.structure)
    
    def get_site_fingerprint(self):
        each_motif_site_print = []
        for site in range(len(self.final_el_list)):
            motif_site_print = get_site_fingerprints(self.structure)[site]
            site_print = [v for k, v in motif_site_print.items()]
            each_motif_site_print.append(site_print)
        return each_motif_site_print
    
    def get_all_motif_data(self):
        comp_motif = {}
        try:
            coord_env_list, motif_type_list = self.get_motif_details()
            comp_motif[self.filename] = {
                'compositions': self.get_motif_comp()[0],
                'motif_type': motif_type_list,
                'csm_details': coord_env_list,
                'structure_fingerprint': self.get_structure_fingerprint(),
                'site_fingerprint': self.get_site_fingerprint()
            }
        except Exception as e:
            error_msg = f"An error occurred for filename {self.filename}: {e}"
            logging.error(error_msg)
            print(error_msg)
        return comp_motif

def list_structure_files(folder_path, file_extension='.cif'):
    structure_files = []
    for file in os.listdir(folder_path):
        if file.endswith(file_extension):
            structure_files.append(os.path.join(folder_path, file))
    return structure_files

def process_structure_file(structure_file):
    motif_extractor = MotifFeaturesDetails(structure_file)
    return motif_extractor.get_all_motif_data()

folder_path = '/home/anoj/work/network/data/oxide_cif_files/'
structure_files = list_structure_files(folder_path)
if __name__ == "__main__":
    structure_files = structure_files
    start_time = time.time()

    results = []
    for file in structure_files:
        result = process_structure_file(file)  # Assuming process_structure_file is defined elsewhere
        results.append(result)

    end_time = time.time()
    total_time = end_time - start_time
    print("Total execution time:", total_time, "seconds")

    with open("./motif_data_all2.pkl", "wb") as f:
        pickle.dump(results, f)
