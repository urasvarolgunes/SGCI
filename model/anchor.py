'''
Graph semi-supervised learning via anchor-graph. 
The idea is to sample a bunch of points from the dataset,
do K-nearest-neighbor search from any point to them, 
construct the adjacency matrix. Hopefully this is as good as 
the graph based on the whole set. 
Shibo Yao, Jan 7 2021

use the final 0-1 matrix for downstream GCN
can convert lil_matrix to csr_matrix
'''
import numpy as np
import scipy.sparse as sp
import multiprocessing as mp
from hpc import multicore_nnls
import sys
import time


def getAnchorIndex(n, m):
    '''
    n: number of points in data
    m: number of anchors
    return a np array of size m indexing the anchors in the data matrix
    '''
    return np.sort(np.random.choice(n,m,replace=False))


def dis_base(pid, sub_list, x, anchor_index, return_dic):
    '''
    the base for Euclidean distance matrix calculation
    pid: process ID, index: matrix row index, return_dic: result container
    x: feature matrix being shared by all processes
    anchorIndex: index for anchors (of size m)
    '''
    n = len(sub_list)
    m = len(anchor_index)
    small_dis = np.zeros([n,m])
    for i in range(n):
        vec = x[sub_list[i]]
        small_dis[i] = [np.linalg.norm(vec-x[j]) for j in anchor_index]

    return_dic[pid] = small_dis


def distanceAnchorEuclidean(x, Q_index, anchor_index, n_jobs=-1, func=dis_base):
    '''
    multiprocessing Euclidean distance to anchors calculation
    x: feature matrix, anchor_index: index of anchors
    n_jobs: number of jobs, default number of processes on CPU
    '''
    total_cpu = mp.cpu_count()
    if type(n_jobs) is not int or n_jobs < -1 or n_jobs > total_cpu:
        print("Specify correct job number!")
        exit(0)
    elif n_jobs==-1:
        n_jobs = total_cpu

    index_list = np.array_split(Q_index, n_jobs)
    processes = []
    return_dic = mp.Manager().dict()

    for i in range(n_jobs):
        proc = mp.Process(target=func, args=(i,index_list[i],x,anchor_index,return_dic))
        processes.append(proc)
        proc.start()
    for process in processes:
        process.join()

    dis_mat = [return_dic[i] for i in range(n_jobs)]
    dis_mat = np.concatenate(dis_mat, axis=0)
    
    return dis_mat


def knn_base(pid, sub_list, anchor_index, dis, k, return_dic):
    '''
    search for k-nearest-neighbors, return zero one, 
    only the search space is reduced from the full set to the anchors
    '''
    n = dis.shape[0]
    r = len(sub_list)
    mat = sp.lil_matrix((n,n),dtype=int)
    for i in range(r): #p
        index = np.argsort(dis[sub_list[i]])[:(k+1)]
        #index = np.argsort(dis[sub_list[i]])[-(k+1):]
        if sub_list[i] == anchor_index[index[0]]:
            index = index[1:(k+1)]
        else:
            index = index[:k]
        mat[sub_list[i], anchor_index[index]] = 1

    return_dic[pid] = mat


def anchorKNN(dis, k, Q_index, anchor_index, n_jobs=-1, func=knn_base):
    if (k > dis.shape[1]-1):
        print("k cannot exceed m-1")
        exit(1)

    total_cpu = mp.cpu_count()
    if type(n_jobs) is not int or n_jobs < -1 or n_jobs > total_cpu:
        print("Specify correct job number!")
        exit(0)
    elif n_jobs==-1:
        n_jobs = total_cpu

    sub_list = np.array_split(Q_index, n_jobs)
    processes = []
    return_dic = mp.Manager().dict()

    for i in range(n_jobs):
        proc = mp.Process(target=func, args=(i,sub_list[i],anchor_index,dis,k,return_dic))
        processes.append(proc)
        proc.start()
    for process in processes:
        process.join()

    graph_knn = [return_dic[i] for i in range(n_jobs)]
    graph_knn = sum(graph_knn)
    
    return graph_knn


if __name__ == "__main__":
    n = int(sys.argv[1]) #number of nodes / datapoints
    p = int(sys.argv[2]) #number of nodes with known label
    d = int(sys.argv[3]) #dimensionality
    m = int(sys.argv[4]) #number of anchors, m <= n-1
    k = int(sys.argv[5]) #k in knn, k <= m
    # run test as "python anchor.py 10000 1000 200 300 20"
    X = np.random.randn(n,d)

    Q_index = range(n)
    start = time.time()
    #anchor_index = getAnchorIndex(n,m)
    anchor_index = getAnchorIndex(p,m)
    dis = distanceAnchorEuclidean(X, Q_index, anchor_index, n_jobs=-1)
    print("distance mat: %.2f s" % (time.time()-start))
    
    start = time.time()
    G = anchorKNN(dis, k, Q_index, anchor_index, n_jobs=-1)
    print("KNN construction: %.2f s" % (time.time()-start))
    
    start = time.time()
    W = multicore_nnls(X, G, Q_index, n_jobs=1)#careful dealing with large graph
    print("NNLS solving: %.2f s" % (time.time()-start))

    #print(G[0,anchor_index])
