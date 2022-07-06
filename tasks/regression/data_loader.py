import pandas as pd
import scipy.sparse as sp
import torch
import sys
sys.path.append("../../model")
sys.path.append("../../")
from hpc import multicore_dis, MSTKNN, multicore_nnls
from anchor import getAnchorIndex, distanceAnchorEuclidean, anchorKNN
from utils import *
import multiprocessing as mp
from sklearn.model_selection import StratifiedKFold, KFold
import os

def load_data(aff, semantic, delta, spanning=True, n_jobs=-1, num_folds=5, num_train=500, seed=42):
    """Build the adjacency matrix, node features and ids dictionary."""
    total_cpu = mp.cpu_count()
    if type(n_jobs) is not int or n_jobs < -1 or n_jobs > total_cpu:
        print("Specify correct job number!")
        exit(0)
    elif n_jobs==-1:
        n_jobs = total_cpu
    
    np.random.seed(seed)
    labels = pd.read_csv("../../data/regression/"+semantic+".txt", delimiter=' ', index_col=0, header=None).values
    features = pd.read_csv("../../data/regression/"+aff+".txt", delimiter=' ', index_col=0, header=None).values

    idx = list(range(len(labels))) # sample 5000 words out of 50,000.
    np.random.shuffle(idx)
    idx = idx[:5000]
    idx_train = list(range(num_train)) # pick num_train words for the train set
    idx_test = list(range(num_train,5000))

    labels, features = labels[idx], features[idx] #subsampling

    graph_path = './saved_graphs/adj_{}_{}_{}_seed{}.npy'.format(semantic, aff, delta, seed)
    if not os.path.exists(graph_path):
        print('generating graph, may take a while...')
        Q_index = range(features.shape[0])
        dis = multicore_dis(features, Q_index, n_jobs)
        graph = MSTKNN(dis, Q_index, delta, n_jobs, spanning)
        adj = multicore_nnls(features, graph, Q_index, n_jobs, epsilon=1e-1)
        adj = normalize(adj + sp.eye(adj.shape[0]))
        np.save(graph_path, adj)
    else:
        adj = np.load(graph_path, allow_pickle=True).item()
    
    folds_dict = get_kfold_split(X=features[idx_train], k=num_folds) #X is just a placeholder
    
    return adj, features, labels, idx_test, folds_dict


def get_kfold_split(X, k=5):    
    kf = KFold(n_splits=k, shuffle=False)
    folds_dict = dict()
    for fold, (train_idx, valid_idx) in enumerate(kf.split(X)):
        folds_dict[fold] = [train_idx, valid_idx]
    
    return folds_dict