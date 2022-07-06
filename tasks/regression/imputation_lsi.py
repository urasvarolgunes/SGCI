import numpy as np
import sys 
sys.path.append("../../model")
from lsi import iterSolveQ
from data_loader import load_data
from sklearn.metrics import mean_squared_error as mse


if __name__ == '__main__':
    if (len(sys.argv) != 5): 
        print("Specify base, aff, delta, seed, anchor and m!")
        exit(0)

    basename = sys.argv[1]
    affname = sys.argv[2]
    delta = int(sys.argv[3])
    seed = int(sys.argv[4])
    np.random.seed(seed)
    
    adj, features, labels, idx_test, folds_dict = load_data(affname, basename, delta, num_train=500, seed=seed)
    idx_train = list(range(len(labels)-len(idx_test)))
    PQ = iterSolveQ(labels[idx_train], adj, 1e-4)
    print("MSE: ", mse(labels[idx_test], PQ[idx_test]))