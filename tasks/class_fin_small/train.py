import time
import argparse
import numpy as np

import torch
import torch.optim as optim
import torch.nn as nn

from sklearn.neighbors import KNeighborsClassifier

import sys
sys.path.append("../../model")
from models import GCN
from propagation import PPRPowerIteration
from data_loader import load_data
import pdb
import math
import optuna
import json

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False,
                    help='Validate during training pass.')
parser.add_argument('--seed', type=int, default=1, help='Random seed.')
parser.add_argument('--epochs', type=int, default=1000,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.001,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=600,
                    help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--batch_size', type=int, default=400, help='batch size')
parser.add_argument('--p', type=int, default=2,
                            help='Number of propagation.')
parser.add_argument('--alpha', type=float, default=0.0,
                            help='Teleport strength.')
parser.add_argument('--delta', type=int, default=8,
                    help='node degree setting in MST-KNN graph')
parser.add_argument('--base', type=str, default='google',
                    help='base embedding: google, glove, fast')
parser.add_argument('--aff', type=str, default='aff',
                    help='affinity info: aff, google, glove, fast')
parser.add_argument('--model', type=str, default='SGC',
                    choices=['MLP', 'SGC'])
parser.add_argument('--num_folds', type=int, default=5)
parser.add_argument('--n_trials', type=int, default=2)

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

def train(model, optimizer, epoch, idx_train, features, labels):
    model.train()
    optimizer.zero_grad()
    mse_loss = nn.MSELoss(reduction = "mean")
    permutation = torch.randperm(idx_train.size()[0])
    indices = idx_train[permutation]
    output = model(features, indices)
    
    loss_train = mse_loss(output, labels[indices])
    loss_train.backward()
    optimizer.step()
    
    return loss_train.item()

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

def test(model, features, labels, idx_test):
    mse_loss = nn.MSELoss(reduction = "mean")
    model.eval()
    with torch.no_grad():
        pred = model(features, idx_test)
        test_loss = mse_loss(pred, labels[idx_test]).item()
    
    return test_loss

def test_KNN(model, features, labels, y, idx_test, idx_train=None):
    '''to test for train+valid set, provide idx_train and set idx_test=idx_valid''' 
    model.eval()
    with torch.no_grad():
        pred = model(features, idx_test)
        N = [2,5,8,10,15,20,30]
        if idx_train is None: #evaluating on train+val+test set
            X = torch.cat((labels, pred), dim=0)
        else: #evaluating on train+val set
            X = torch.cat((labels[idx_train], pred), dim=0)
            y = y[torch.cat([idx_train, idx_test])] #idx_test=idx_valid in this case
        
        X = X.cpu().detach().numpy()

        result = []
        for n in N: # classification n_neighbors = 5, n_components = 30 up
            result.append(KNN(X.copy(), y.copy(), n))
    
    return result


def show_params(args):
    print("hidden: ", args.hidden)
    print("epochs: ", args.epochs)
    print("p: ", args.p)
    print("alpha: ", args.alpha)
    print("base: ", args.base)
    print("delta: ", args.delta)


def run_training(fold, params, use_all_train=False, test_knn=False):
    '''runs training for one on the folds'''
    result = {n:[] for n in [2,5,8,10,15,20,30]}
    
    adj, features, labels, y, folds_dict = load_data(args.aff, args.base, args.delta, num_folds=args.num_folds)
    features = torch.FloatTensor(features)
    labels = torch.FloatTensor(labels)
    #adj = sparse_mx_to_torch_sparse_tensor(adj)
        
    idx_train = torch.LongTensor(folds_dict[fold][0])
    idx_valid = torch.LongTensor(folds_dict[fold][1])
    if use_all_train:
        idx_train = torch.cat([idx_train, idx_valid])
    idx_test = torch.LongTensor(np.array(range(labels.shape[0], features.shape[0])))
    
    propagator = PPRPowerIteration(adj, params['alpha'], params['p'])

    # Set up the GCN Model and the optimizer
    model = GCN(nfeatures=features.shape[1],
                hiddenunits=[args.hidden],
                nout=labels.shape[1],
                drop_prob=params['dropout'],
                propagation=propagator)
    optimizer = optim.Adam(model.parameters(), lr=params['lr'], weight_decay=params['wd'])

    if args.cuda:
        model.cuda()
        features = features.cuda()
        labels = labels.cuda()
        #adj = adj.cuda()
        idx_train = idx_train.cuda()
        idx_valid = idx_valid.cuda()
    
    for epoch in range(params['epochs']):
        train_loss = train(model, optimizer, epoch, idx_train, features, labels)    
    
    if not use_all_train:        
        val_knn_result = test_KNN(model, features, labels, y, idx_test=idx_valid, idx_train=idx_train)
        best_loss = -np.mean(val_knn_result)
    else:
        best_loss = None
    
    test_knn_result = None
    
    if test_knn:
        test_knn_result = test_KNN(model, features, labels, y, idx_test=idx_test, idx_train=None)
            
    return best_loss, test_knn_result

def objective(trial):
        
    if args.model == 'MLP':
        params={'lr':3e-4,
                'wd':trial.suggest_loguniform("wd", 1e-7, 1e-4),
                'dropout':trial.suggest_uniform("dropout", 0.0, 0.7),
                'alpha': 0,
                'p': 0,
                'epochs':trial.suggest_int('epochs', 75, 500, step=25)
               }
    elif args.model == 'SGC':
        params={'lr':3e-4,
                'wd':trial.suggest_loguniform("wd", 1e-7, 1e-4),
                'dropout':trial.suggest_uniform("dropout", 0.0, 0.7),
                'alpha': 0,
                'p': 2,
                'epochs':trial.suggest_int('epochs', 100, 500, step=25)
               }
        
    test_results = []
    all_losses = []
    
    for fold in range(args.num_folds):
        temp_loss, result = run_training(fold, params)
        all_losses.append(temp_loss)
        test_results.append(result)
    
    return np.mean(all_losses)


if __name__ == "__main__":
 
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
          
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=args.n_trials)
    
    print("best_trial:")
    best_trial = study.best_trial
    
    print(best_trial.values)
    print(best_trial.params)    
    
    
    test_results = []
    test_params = best_trial.params
    test_params['lr'] = 3e-4
    test_params['alpha'] = 0
    
    if args.model == 'MLP':
        test_params['p'] = 0
    elif args.model == 'SGC':
        test_params['p'] = 2
    

    for fold in range(args.num_folds):
        _, result = run_training(fold, test_params, use_all_train=True, test_knn=True)
        test_results.append(result)
    
    print(f'KNN TEST RESULTS FOR {args.model}, {args.base}')
    print('mean:', np.mean(test_results, axis=0))
    print('std:', np.std(test_results ,axis=0, ddof=1))