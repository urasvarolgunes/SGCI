import numpy as np
import sys
sys.path.append("../../model")
from lsi import iterSolveQ
from data_loader import load_data
from sklearn.neighbors import KNeighborsClassifier


def KNN(X, y, n):
    l = len(y)
    y_hat = []
    for i in range(l):
        X_train = np.delete(X, i, axis = 0)
        y_train = np.delete(y, i, axis = 0)
        neigh = KNeighborsClassifier(n_neighbors = n)
        neigh.fit(X_train, y_train)
        y_hat.extend(neigh.predict(X[i].reshape(1,-1)))
    acc = sum(np.array(y_hat) == y) / l 
    return acc 


def test(X, y):
    N = [2,5,8,10,15,20,30]
    result = []
    for n in N: # classification n_neighbors = 5, n_components = 30 up
        result.append(KNN(X.copy(), y.copy(), n))
    return result


if __name__ == '__main__':
    if (len(sys.argv) != 5):
        print("Specify base, aff delta and seed!")
        exit(0)

    basename = sys.argv[1]
    affname = sys.argv[2]
    delta = int(sys.argv[3])
    np.random.seed(int(sys.argv[4]))

    #adj, features, labels, y, idx_train = load_data_lsi(affname, basename, delta)
    adj, features, labels, y, folds_dict = load_data(affname, basename, delta)
    idx_train = list(range(len(labels)))
    PQ = iterSolveQ(labels[idx_train], adj, 1e-3)
    print(test(PQ, y), '\n')
