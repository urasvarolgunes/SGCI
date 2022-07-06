import pandas as pd
import numpy as np
import scipy.sparse as sp
import sys
sys.path.append("../../../model")
sys.path.append("../../")
from hpc import multicore_dis, MSTKNN, multicore_nnls
from utils import *
import multiprocessing as mp
from sklearn.model_selection import StratifiedKFold
import os

def load_data(aff, semantic, delta, ita=1e-4, spanning=True, n_jobs=-1, num_folds=5):
    """Build the adjacency matrix, node features and ids dictionary."""
    total_cpu = mp.cpu_count()
    if type(n_jobs) is not int or n_jobs < -1 or n_jobs > total_cpu:
        print("Specify correct job number!")
        exit(0)
    elif n_jobs==-1:
        n_jobs = total_cpu

    labels = pd.read_csv("../../data/fin_small/"+semantic+"Mat.csv", index_col=0)
    labels= labels.iloc[:,:-2]#last 2 cols are frequency and sector label
    features = pd.read_csv("../../data/fin_small/"+aff+"Mat.csv", index_col=0)
    features, labels, semanticOuter = permuteMat(features, labels)
    
    y = features.y.values   
    features = features.iloc[:,:-1]
    
    graph_path = './saved_graphs/adj_{}_{}.npy'.format(semantic, delta)
    if not os.path.exists(graph_path):
        print('generating graph')
        Q_index = range(features.shape[0])
        dis = multicore_dis(features.values, Q_index, n_jobs)
        graph = MSTKNN(dis, Q_index, delta, n_jobs, spanning)
        adj = multicore_nnls(features.values, graph, Q_index, n_jobs, epsilon=1e-1)

        idx_train = np.array(range(labels.shape[0]))
        adj = normalize(adj + sp.eye(adj.shape[0]))
        np.save(graph_path, adj)
    
    adj = np.load(graph_path, allow_pickle=True).item()
    
    features = features.values
    labels = labels.values
    idx_train = np.array(range(labels.shape[0]))
    
    if not os.path.exists('./{}_{}_fold_dict.npy'.format(semantic, num_folds)):
        folds_dict = get_kfold_split(X=features[idx_train] ,y=y[idx_train], k=num_folds) #X is just a placeholder
        np.save('{}_{}_fold_dict.npy'.format(semantic, num_folds), folds_dict)
    
    folds_dict = np.load('{}_{}_fold_dict.npy'.format(semantic, num_folds), allow_pickle=True).item()
        
    return adj, features, labels, y, folds_dict


def get_kfold_split(X, y, k=5):    
    skf = StratifiedKFold(n_splits=k, shuffle=True)
    folds_dict = dict()
    for fold, (train_idx, valid_idx) in enumerate(skf.split(X, y)):
        folds_dict[fold] = [train_idx, valid_idx]
    
    return folds_dict
