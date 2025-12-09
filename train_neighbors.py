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

"""
def initialize_embeddings(data, node_u, node_v, embedding_dim):
    #Initialize embeddings for materials and motifs with dimensionality reduction.
    node_list_u, node_list_v = {}, {}

    def reduce_dimensionality(vectors, input_dim, output_dim):
        array = np.array(vectors)  # Ensures vectors is a continuous block of memory
        tensor = torch.tensor(array, dtype=torch.float32)
        linear_layer = nn.Linear(input_dim, output_dim)
        activation = nn.ReLU()
        reduced = activation(linear_layer(tensor))
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
def initialize_embeddings(data, node_u, node_v, atomic_dict, embedding_dim=128):
    #Initialize embeddings for materials and motifs with normalization before padding.
    node_list_u, node_list_v = {}, {}

    def normalize_and_pad_vector(vector, target_dim):
        # L2 normalization
        norm = np.linalg.norm(vector) + 1e-10
        normalized_vector = vector / norm

        # Padding
        if len(normalized_vector) < target_dim:
            padding = np.zeros(target_dim - len(normalized_vector))
            normalized_vector = np.concatenate((normalized_vector, padding))

        return normalized_vector

    # Initialize material embeddings (u nodes)
    for node in node_u:
        comp = data[node[1:]]['formula']
        comp_vector = vectorize_composition(comp)
        cgcnn_vector = get_cgcnn_el_vectors(comp, atomic_dict)
        padded_vector = normalize_and_pad_vector(cgcnn_vector, embedding_dim)
        node_list_u[node] = {
            'embedding_vectors': padded_vector.reshape(1, embedding_dim), 
            'context_vectors': padded_vector.reshape(1, embedding_dim)
        }

    # Initialize motif embeddings (v nodes)
    for node in node_v:
        motif = node.split('__')[0][1:]
        motif_type = node.split('__')[-1]
        motif_vector = encode_motif_type(motif_type)
        padded_vector = normalize_and_pad_vector(motif_vector, embedding_dim)
        node_list_v[node] = {
            'embedding_vectors': padded_vector.reshape(1, embedding_dim), 
            'context_vectors': padded_vector.reshape(1, embedding_dim)
        }


    return node_list_u, node_list_v


def get_first_order_nbrs(graph, node_list):
    """
    Efficiently find 2-hop neighbors of the same type in a bipartite graph.
    
    Arguments:
    - graph: bipartite graph with 'material' and 'motif' node types
    - node_list: nodes of the same bipartite type (e.g., materials only)

    Returns:
    - dict mapping node -> list of 2-hop neighbors of the same type
    """
    node_set = set(node_list)
    context = {}

    for node in node_list:
        nbrs = set(graph.neighbors(node))
        first_order_neighbors = set()
        for inter in nbrs:
            # second-order neighbors of original node
            second_neighbors = set(graph.neighbors(inter))
            first_order_neighbors.update(second_neighbors & node_set)
        first_order_neighbors.discard(node)
        context[node] = list(first_order_neighbors)

    return context


"""
def get_contexts_from_orig_G(graph, node_list, max_nbr_order=4, max_neighbors=50):

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

"""
def stable_sigmoid(X):
    if X >= 0:
        return 1 / (1 + np.exp(-X))
    else:
        exp_X = np.exp(X)
        return exp_X / (1 + exp_X)


def KL_divergence( u, v, w, node_list_u, node_list_v, lam, gamma, data):
    """
    KL divergence update:
    - u: material node
    - v: motif node (shared globally across materials, full ID e.g., 'AO6__Octahedral')
    - data[u][v]: contains site_fp specific to material-motif pair
    """
    loss = 0
    e_ij = w

    U = np.array(node_list_u[u]['embedding_vectors'])  # shape (1, d)
    V = np.array(node_list_v[v]['embedding_vectors'])  
    U_norm = U / (np.linalg.norm(U) + 1e-10)
    V_norm = V / (np.linalg.norm(V) + 1e-10)
    X = np.dot(U_norm, V_norm.T)
    sigmoid = 1.0 / (1.0 + np.exp(-np.clip(X, -709, 709)))
    sigmoid_scalar = sigmoid.flatten()[0]

    grad = gamma * lam * (e_ij * (1 - sigmoid_scalar))
    # Material update: pulled toward structural representation V
    update_u = grad * V
    
    # Get site_fp vector for this material-motif pair
    try:
        site_fp = np.array(data[u][v]['site_fp'])
        if site_fp.shape[0] < U.shape[1]:
            site_fp = np.concatenate([site_fp, np.zeros(U.shape[1] - site_fp.shape[0])])
        else:
            site_fp = site_fp[:U.shape[1]]
        site_fp /= (np.linalg.norm(site_fp) + 1e-10)
    except KeyError:
        site_fp = np.zeros_like(U)

    # Motif update: influenced by U and site_fp jointly
    blend = 0.5  # can be tuned. 
    update_v = grad * (blend * U + (1 - blend) * site_fp.reshape(1, -1))

    try:
        loss += gamma * e_ij * math.log(sigmoid_scalar + 1e-8)
    except ValueError:
        pass

    return update_u, update_v, loss

"""
def KL_divergence( u, v, w, node_list_u, node_list_v, lam, gamma):
    loss = 0
    e_ij = w 
    #print(e_ij)

    U = np.array(node_list_u[u]['embedding_vectors'])
    V = np.array(node_list_v[v]['embedding_vectors'])
    try:
        site_fp = np.array(data[u][v]['site_fp'])
        if site_fp.shape[0] < U.shape[1]:
            site_fp = np.concatenate([site_fp, np.zeros(U.shape[1] - site_fp.shape[0])])
        else:
            site_fp = site_fp[:U.shape[1]]
        site_fp /= (np.linalg.norm(site_fp) + 1e-10)
    except KeyError:
        site_fp = np.zeros_like(U)
    X = np.dot(U, site_fp.T)
    sigmoid = 1.0 / (1.0 + np.exp(-np.clip(X, -709, 709)))

    # Ensure sigmod is scalar
    sigmoid_scalar = sigmoid.flatten()[0]

    update_v = grad * ( V + site_fp.reshape(1, -1))
    update_v = gamma * lam * (e_ij * (1 - sigmoid_scalar)) * U

    try:
        loss += gamma * e_ij * math.log(sigmoid_scalar)
    except ValueError:
        pass

    return update_u, update_v, loss
"""

def skip_gram(center, contexts, negs, node_list, lam, pa):
    loss = 0
    I_z = {center: 1}  # Indication function
    for node in negs:
        I_z[node] = 0

    V = np.array(node_list[contexts]['embedding_vectors'])
    update = np.zeros_like(V)

    for u in I_z.keys():
        if u not in node_list:
            continue
        Theta = np.array(node_list[u]['context_vectors'])
        cos_sim = cosine_similarity(V, Theta)
        normalized_cos_sim = (1 + cos_sim) / 2
        gradient = pa * lam * (I_z[u] - normalized_cos_sim) 
        scaled_update = gradient * Theta
        # Update
        update+= scaled_update
        if np.linalg.norm(update) > 0:
            update /= np.linalg.norm(update)
        node_list[u]['context_vectors']+= update
        
        epsilon = 1e-10
        try:
            loss += pa * (I_z[u] * np.log(normalized_cos_sim + epsilon) + (1 - I_z[u]) * np.log(1 - normalized_cos_sim + epsilon))
        except ValueError:
            pass

    return update, loss

# -------------------- Main Training Script --------------------

def main():
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('--train-data', default='../data/edges_train.dat')
    #parser.add_argument('--test-data', default='../data/edges_test.dat')
    parser.add_argument('--model-path', default='../data/')
    parser.add_argument('--model-name', default='default')
    parser.add_argument('--d', default=128, type=int, help='Embedding size.')
    parser.add_argument('--max-iter', default=500, type=int, help='Max iterations.')
    parser.add_argument('--alpha', default=0.01, type=float, help='Alpha parameter.')
    parser.add_argument('--beta', default=0.01, type=float, help='Beta parameter.')
    parser.add_argument('--gamma', default=0.1, type=float, help='Gamma parameter.')
    parser.add_argument('--lam', default=0.01, type=float, help='Learning rate.')
    parser.add_argument('--nbr-order', default=4, type=int, help='Order for same type neighbors')
    parser.add_argument('--neg-order', default=8, type=int, help='Order for negatives neighbors.')
    parser.add_argument('--save-interval', default=50, type=int, help='Save embeddings every N iterations.')
    args = parser.parse_args()
    
    print("---Reading data---")

    #MP-perovskites
    with open("/home/anoj/work/network/motif_data/mp_perovskites_data.pkl", 'rb') as f:
        data = pickle.load(f)
    with open("/home/anoj/work/network/motif_data/mp_perovskites_motifops_data.pkl", 'rb') as f:
        data_motif = pickle.load(f)
    """
    #Perovskites
    with open("/home/anoj/work/network/motif_data/matbench_perovskites_data.pkl", 'rb') as f:
        data = pickle.load(f)
    with open("/home/anoj/work/network/motif_data/matbench_perovskites_motifops_data.pkl", 'rb') as f:
        data_motif = pickle.load(f)

    """
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
    
    node_list_u, node_list_v = initialize_embeddings(data = data, node_u=gul.node_u, node_v=gul.node_v, atomic_dict = atomic_dict_cgcnn, embedding_dim = args.d)
    node_init_u , node_init_v = node_list_u, node_list_v
    
    embeddings_dir = os.path.join(args.model_path, 'perovs_nbrs1')
    os.makedirs(embeddings_dir, exist_ok=True)
    save_embeddings(node_list_u, os.path.join(embeddings_dir, 'vectors_u_initial.dat'))
    save_embeddings(node_list_v, os.path.join(embeddings_dir, 'vectors_v_initial.dat'))
    
    print("---Initial embeddings generated---")
    
    print("---Neighbor search---")
    context_u = get_first_order_nbrs(gul.G, gul.node_u)
    context_v = get_first_order_nbrs(gul.G, gul.node_v)
    print("---Materials and Motifs contexts done---")
    
    
    # Train model
    print("---Training---")
    alpha, beta, gamma, lam  = args.alpha, args.beta, args.gamma, args.lam 
    print('======== experiment settings =========')
    print('alpha : %0.4f, beta : %0.4f, gamma : %0.4f, lam : %0.4f, d : %d' % (alpha, beta, gamma, lam, args.d))
    
    last_loss = float('inf')
    for iteration in range(args.max_iter):
        print(f"Iteration {iteration+1} of {args.max_iter}")
        print(f"Progress: {((iteration+1) / args.max_iter) * 100:.2f}%")
        loss = 0
        random.shuffle(gul.edge_list)
        for u, v, w in gul.edge_list:
            #print(u, v, w)
            # Skip-gram updates
            context_u_set = context_u[u] if u in context_u else []
            context_v_set = context_v[v] if v in context_v else []
            neg_u, neg_v = [], []
            for index in range(len(context_u_set)):
                #for z1 in context_u_set[index]:
                z1 = context_u_set[index]
                #print("z1", z1)
                update, tmp_loss = skip_gram(u, z1, neg_u, node_list_u, lam, alpha)
                node_list_u[z1]['embedding_vectors'] += update
                loss += tmp_loss
#            for index in range(len(context_v_set)):
#                #for z2 in context_v_set[index]:
#                z2 = context_v_set[index]
#                #print("z2", z2)
#                update, tmp_loss = skip_gram(v, z2, neg_v, node_list_v, lam, beta)
#                node_list_v[z2]['embedding_vectors'] += update
#                loss += tmp_loss
            # KL-divergence update
            update_u, update_v, tmp_loss = KL_divergence(u, v, w, node_list_u, node_list_v, lam, gamma, data = data_motif)
            node_list_u[u]['embedding_vectors'] += update_u
            node_list_v[v]['embedding_vectors'] += update_v
            loss += tmp_loss

        # Save embeddings every N iterations
        if (iteration + 1) % args.save_interval == 0:
            save_embeddings(node_list_u, os.path.join(embeddings_dir, f'vectors_u_iter{iteration + 1}.dat'))
            save_embeddings(node_list_v, os.path.join(embeddings_dir, f'vectors_v_iter{iteration + 1}.dat'))

        # Early stopping 
        delta_loss = abs(loss - last_loss)
        if last_loss > loss:
            lam *= 1.05
        else:
            lam *= 0.95
        last_loss = loss
        if delta_loss < 1e-4:
            break  

    # Save embeddings
    print(f"Delta Loss: {delta_loss}")
    save_embeddings(node_list_u, os.path.join(embeddings_dir, 'vectors_u_final.dat'))
    save_embeddings(node_list_v, os.path.join(embeddings_dir, 'vectors_v_final.dat'))
    print("---Done---")

if __name__ == "__main__":
    main()
