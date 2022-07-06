import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def normalize(mx):
    """Row-normalize sparse matrix."""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

def row_normalize(adj):
    """row normalize"""
    adj = sp.coo_matrix(adj)
    degree = np.array(adj.sum(1)).flatten()
    d_mat_inv = sp.diags(1./np.maximum(degree, np.finfo(float).eps))
    return d_mat_inv.dot(adj).tocoo()

def sym_normalize_adj(adj):
    """symmetrically normalize adjacency matrix"""
    adj = sp.coo_matrix(adj)
    degree = np.array(adj.sum(1)).flatten()
    d_inv_sqrt = np.power(np.maximum(degree, np.finfo(float).eps), -0.5)
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()

def preprocess_high_order_adj(adj, order, eps):
    """A higher-order polynomial with sparsification"""
    adj = row_normalize(adj)
    adj_sum = adj
    cur_adj = adj
    for i in range(1, order):
        cur_adj = cur_adj.dot(adj)
        adj_sum += cur_adj
    adj_sum /= order

    adj_sum.setdiag(0)
    adj_sum.data[adj_sum.data<eps] = 0
    adj_sum.eliminate_zeros()

    adj_sum += sp.eye(adj.shape[0])
    return sym_normalize_adj(adj_sum + adj_sum.T)

def Linv(adj):
    """inverse shifted Laplacian filter"""
    L = sp.csgraph.laplacian(adj)
    adj = sp.linalg.inv(sp.csc_matrix(L+sp.eye(L.shape[0])))
    return adj

def permuteMat(aff, semantic): #permute affinity mat and semantic mat
    affInd = aff.index.tolist()
    semanticInd = semantic.index.tolist() #instead of index.values.tolist()
    Pind = [i for i in semanticInd if i in affInd]
    Qind = [i for i in affInd if i not in Pind]

    PMat = aff.loc[Pind].copy()
    QMat = aff.loc[Qind].copy()

    aff = pd.concat([PMat, QMat], axis=0)
    semanticInter = semantic.loc[Pind].copy()
    semanticOuter = semantic.drop(labels=Pind, axis=0)

    return aff, semanticInter, semanticOuter

def poly_adj(adj, order):
    """A higher-order polynomial"""
    adj_sum = sp.eye(adj.shape[0])
    cur_adj = adj 
    for i in range(order):
        adj_sum += cur_adj
        cur_adj = cur_adj.dot(adj)

    return normalize(adj_sum)
