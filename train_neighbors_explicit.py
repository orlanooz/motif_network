#!/usr/bin/env python
#!/usr/bin/env python



from argparse import ArgumentParser, FileType, ArgumentDefaultsHelpFormatter
import sys, os, re, math, datetime, random

import json, torch, pickle
import numpy as np
import pandas as pd
import torch.nn as nn
import networkx as nx

from sklearn import preprocessing
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score,auc,precision_recall_fscore_support

from collections.abc import Iterable
from collections import defaultdict

from pymatgen.core.composition import Composition
from pymatgen.core.periodic_table import Element

from data_utils import DataUtils
from graph_utils import GraphUtils
from motif_type_info import most_occ_envs, motif_type_nmbrs



# -------------------- Utility Functions --------------------

def vectorize_composition(composition):
    """Convert material composition into a one-hot encoded vector."""
    onehot_vector = np.zeros(118)
    for element in Composition(composition):
        onehot_vector[element.Z - 1] = 1
    return onehot_vector

def get_megnet_el_vectors(composition, atomic_dict):
    """Retrieve elemental vectors from atomic data."""
    elem_vector_summed = np.zeros_like(next(iter(atomic_dict.values())))
    for element in Composition(composition):
        atomic_vector = np.array(atomic_dict[str(Element(element))])
        elem_vector_summed += atomic_vector
    return elem_vector_summed.flatten()

def get_cgcnn_el_vectors(composition, atomic_dict):
    """Retrieve elemental vectors from atomic data."""
    elem_vector_summed = np.zeros_like(next(iter(atomic_dict.values())))
    for element in Composition(composition):
        atomic_vector = np.array(atomic_dict[str(Element(element).Z)])
        elem_vector_summed += atomic_vector
    return elem_vector_summed.flatten()

def encode_motif_type(motif_type):
    """Encode motif type into a one-hot vector."""
    one_hot = np.zeros(65)
    if motif_type in motif_type_nmbrs:
        one_hot[motif_type_nmbrs[motif_type] - 1] = 1
    return one_hot

def cosine_similarity(x, y):
    """Calculate cosine similarity between two vectors."""
    return np.dot(x, y.T) / (np.linalg.norm(x) * np.linalg.norm(y))

def save_embeddings(node_list, file_path):
    """Save node embeddings to a file."""
    with open(file_path, "w") as f:
        for node, data in node_list.items():
            embedding_str = " ".join(map(str, data['embedding_vectors'].flatten()))
            f.write(f"{node} {embedding_str}\n")


# -------------------- Graph and Sampling Functions --------------------
import torch
import numpy as np
import torch.nn as nn
def initialize_embeddings(data, node_u, node_v, embedding_dim):
    """Initialize embeddings for materials and motifs with dimensionality reduction."""
    node_list_u, node_list_v = {}, {}

    def reduce_dimensionality(vectors, input_dim, output_dim):
        tensor = torch.tensor(vectors, dtype=torch.float32)
        linear_layer = nn.Linear(input_dim, output_dim)
        reduced = linear_layer(tensor)
        return reduced.detach().numpy()

    # Initialize material embeddings
    for node in node_u:
        comp = data[node[1:]]['formula']
        vector = vectorize_composition(comp)
        input_dim = len(vector)
        if input_dim != embedding_dim:
            reduced_vector = reduce_dimensionality([vector], input_dim, embedding_dim)[0]
        else:
            reduced_vector = vector
        node_list_u[node] = {
            'embedding_vectors': reduced_vector.reshape(1, embedding_dim),
            'context_vectors': reduced_vector.reshape(1, embedding_dim)
        }

    # Initialize motif embeddings
    for node in node_v:
        motif = node.split('__')[0][1:]
        motif_type = node.split('__')[-1]
        motif_vector = encode_motif_type(motif_type)
        comp_vector = vectorize_composition(motif)
        combined_vector = np.concatenate((motif_vector, comp_vector))
        input_dim = len(combined_vector)
        reduced_vector = reduce_dimensionality([combined_vector], input_dim, embedding_dim)[0]

        node_list_v[node] = {
            'embedding_vectors': reduced_vector.reshape(1, embedding_dim),
            'context_vectors': reduced_vector.reshape(1, embedding_dim)
        }

    return node_list_u, node_list_v


"""
def get_contexts_and_negatives(graph, node_list, nbr_order=4, neg_order=8, max_neighbors=50, max_negatives = 100):
    results = {}
    node_degrees = dict(graph.degree())  # Cache degrees for sorting

    for node in node_list:
        # Initialize result for this node
        results[node] = {"contexts": [], "negatives": []}

        # Find reachable nodes up to `nbr_order` for contexts
        reachable_nodes = nx.single_source_shortest_path_length(graph, node, cutoff=nbr_order)
        included_context = set()

        # Populate contexts as list of lists
        for order in range(1, nbr_order + 1):
            order_neighbors = [
                nbr for nbr, dist in reachable_nodes.items() if dist == order and nbr != node
            ]

            if order == 1:
                # Include all first-order neighbors without limiting
                sorted_neighbors = sorted(order_neighbors, key=lambda n: node_degrees[n], reverse=True)
            else:
                # Limit second-order and higher neighbors
                sorted_neighbors = sorted(order_neighbors, key=lambda n: node_degrees[n], reverse=True)[:max_neighbors]

            included_context.update(sorted_neighbors)
            results[node]["contexts"].append(sorted_neighbors)

        # Find negatives: nodes not reachable within `neg_order`
        reachable_within_neg_order = set(
            nx.single_source_shortest_path_length(graph, node, cutoff=neg_order).keys()
        )
        all_nodes = set(graph.nodes)
        potential_negatives = all_nodes - reachable_within_neg_order - included_context

        # Sort negatives by low degree
        results[node]["negatives"] = sorted(potential_negatives, key=lambda n: node_degrees[n])[:max_negatives]

    return results
"""

def get_contexts_from_orig_G(graph, node_list, max_nbr_order=4, max_neighbors=50):
    """
    Get contexts (neighbors) for each node in the node list from the graph.

    Parameters:
    - graph (nx.Graph): The input graph.
    - node_list (list): List of nodes for which to find contexts.
    - nbr_order (int): Maximum order of neighbors to include in contexts.
    - max_neighbors (int): Maximum number of neighbors for second-order and higher contexts.

    Returns:
    - dict: Contexts for each input node.
            Format: {node: {"contexts": [[order_1_neighbors], [order_2_neighbors], ...]}}
    """
    results = {}
    node_degrees = dict(graph.degree())  # Cache degrees for sorting

    for node in node_list:
        # Initialize result for this node
        results[node] = {"contexts": [], "negatives": []}

        # Find reachable nodes up to `nbr_order` for contexts
        reachable_nodes = nx.single_source_shortest_path_length(graph, node, cutoff=max_nbr_order)

        included_context = set()

        # Populate contexts as list of lists (grouped by order)
        for order in range(1, max_nbr_order + 1):
            nbr_range = order*2
            order_neighbors = [
                nbr for nbr, dist in reachable_nodes.items() if dist == nbr_range and nbr != node
            ]

            if nbr_range == 2:
                # Include all first-order neighbors
                sorted_neighbors = sorted(order_neighbors, key=lambda n: node_degrees[n], reverse=True)
            else:
                # Limit the number of neighbors for higher orders
                sorted_neighbors = sorted(order_neighbors, key=lambda n: node_degrees[n], reverse=True)[:max_neighbors]

            # Add sorted neighbors for the current order
            included_context.update(sorted_neighbors)
            results[node]["contexts"].append(sorted_neighbors)

    return results


def stable_sigmoid(X):
    if X >= 0:
        return 1 / (1 + np.exp(-X))
    else:
        exp_X = np.exp(X)
        return exp_X / (1 + exp_X)

def KL_divergence(edge_dict_u, u, v, node_list_u, node_list_v, lam, gamma):
    loss = 0
    e_ij = edge_dict_u[u].get(v, 0)  # Default to 0 if no direct edge exists
    update_u = 0
    update_v = 0
    U = np.array(node_list_u[u]['embedding_vectors'])
    V = np.array(node_list_v[v]['embedding_vectors'])
    X =np.dot(U, V.T).item()
    
    sigmod = stable_sigmoid(X)
    

    update_u += gamma * lam * ((e_ij * (1 - sigmod)) * 1.0 / math.log(math.e, math.e)) * V
    update_v += gamma * lam * ((e_ij * (1 - sigmod)) * 1.0 / math.log(math.e, math.e)) * U

    try:
        loss += gamma * e_ij * math.log(sigmod)
    except ValueError:
        pass

    return update_u, update_v, loss


# -------------------- Main Training Script --------------------

def main():
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('--train-data', default='../data/edges_train.dat')
    #parser.add_argument('--test-data', default='../data/edges_test.dat')
    parser.add_argument('--model-path', default='../data/')
    parser.add_argument('--model-name', default='default')
    parser.add_argument('--d', default=64, type=int, help='Embedding size.')
    parser.add_argument('--max-iter', default=500, type=int, help='Max iterations.')
    parser.add_argument('--alpha', default=0.01, type=float, help='Alpha parameter.')
    parser.add_argument('--beta', default=0.01, type=float, help='Beta parameter.')
    parser.add_argument('--gamma', default=0.1, type=float, help='Gamma parameter.')
    parser.add_argument('--lam', default=0.01, type=float, help='Learning rate.')
    parser.add_argument('--nbr-order', default=2, type=int, help='Order for same type neighbors')
    parser.add_argument('--neg-order', default=8, type=int, help='Order for negatives neighbors.')
    parser.add_argument('--save-interval', default=50, type=int, help='Save embeddings every N iterations.')
    args = parser.parse_args()
    
    print("---Reading data---")
#    with open("/home/anoj/work/network/results/mp_perovs_motif_data.pkl", 'rb') as f:
#        data = pickle.load(f)
    with open("/home/anoj/work/network/results/all_materials_motif_data.pkl", 'rb') as f:
        data = pickle.load(f)
    with open('/home/anoj/work/atom_init.json') as f:
        atomic_dict_cgcnn = json.load(f)
    with open('/home/anoj/work/elemental_embedding_megnet.json') as f:
        atomic_dict_megnet = json.load(f)
    
    # Load data and initialize graph
    print("---Initialize graph---")
    gul = GraphUtils(args.model_name)
    gul.construct_training_graph(args.train_data)
    edge_list = gul.edge_list
    edge_dict_u = gul.edge_dict_u

    #Initial embeddings
    
    node_list_u, node_list_v = initialize_embeddings(data = data, node_u=gul.node_u, node_v=gul.node_v, embedding_dim = args.d)
    
    embeddings_dir = os.path.join(args.model_path, 'embeddings_exp2')
    os.makedirs(embeddings_dir, exist_ok=True)
    save_embeddings(node_list_u, os.path.join(embeddings_dir, 'vectors_u_initial.dat'))
    save_embeddings(node_list_v, os.path.join(embeddings_dir, 'vectors_v_initial.dat'))
    
    print("---Initial embeddings generated---")
    
    #Get contexts as first order neighbors of same type.
    max_order, neg_max_order = args.nbr_order, args.neg_order
    
    context_dict_u = {}
    neg_dict_u = {}
    print("---Materials contexts---")
    outputs_u = get_contexts_from_orig_G(gul.G, gul.node_u, max_nbr_order=max_order, max_neighbors=25)
    for node in gul.node_u:
        context_dict_u[node] = outputs_u[node]['contexts']
        neg_dict_u[node] = outputs_u[node]['negatives']

    print("---Contexts and negatives done for materials---")
    context_dict_v = {}
    neg_dict_v = {}
    #motif_proj_graph = nx.projected_graph(gul.G, gul.node_v)
    print("---Motifs contexts---")
    outputs_v = get_contexts_from_orig_G(gul.G, gul.node_v, max_nbr_order=max_order, max_neighbors=25)
    for node in gul.node_v:
        context_dict_v[node] = outputs_v[node]['contexts']
        neg_dict_v[node] = outputs_v[node]['negatives']

    print("---Contexts and negatives done for motifs---")

    # Train model
    print("---Training---")
    alpha, beta, gamma, lam  = args.alpha, args.beta, args.gamma, args.lam 
    last_loss = float('inf')
    for iteration in range(args.max_iter):
        s1 = "[%s%s]%0.2f%%\n" % ("*" * iteration, " " * (args.max_iter - iteration), iteration * 100.0 / (args.max_iter - 1))
        print(s1, flush=True)
        loss = 0
        
        random.shuffle(gul.edge_list)
        for u, v, w in edge_list:
            # KL-divergence update
            update_u, update_v, tmp_loss = KL_divergence(edge_dict_u, u, v, node_list_u, node_list_v, lam, args.gamma)
            node_list_u[u]['embedding_vectors'] += update_u
            node_list_v[v]['embedding_vectors'] += update_v
            loss += tmp_loss

        # Save embeddings every N iterations
        if (iteration + 1) % args.save_interval == 0:
            save_embeddings(node_list_u, os.path.join(embeddings_dir, f'vectors_u_iter{iteration + 1}.dat'))
            save_embeddings(node_list_v, os.path.join(embeddings_dir, f'vectors_v_iter{iteration + 1}.dat'))

        # Early stopping 
        delta_loss = abs(last_loss - loss)
        if delta_loss < 1e-6: 
            print("Converged.")
            break
        last_loss = loss

    # Save embeddings
    print(delta_loss)
    save_embeddings(node_list_u, os.path.join(embeddings_dir, 'vectors_u_final.dat'))
    save_embeddings(node_list_v, os.path.join(embeddings_dir, 'vectors_v_final.dat'))
    print("---Done---")

if __name__ == "__main__":
    main()
