import time
import argparse
import numpy as np

import torch
import torch.optim as optim
import torch.nn as nn

import sys 
sys.path.append("../../model")
from models import GCN 
from propagation import PPRPowerIteration
from data_loader import load_data
from torch.nn.functional import mse_loss
import optuna

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False, help='Validate during training pass.')
parser.add_argument('--seed', type=int, default=1, help='Random seed.')
parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.001, help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=600, help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0., help='Dropout rate (1 - keep probability).')
parser.add_argument('--batch_size', type=int, default=400, help='batch size')
parser.add_argument('--p', type=int, default=2,help='Number of propagation.')
parser.add_argument('--alpha', type=float, default=0.0, help='Teleport strength.')
parser.add_argument('--delta', type=int, default=10, help='node degree setting in MST-KNN graph')
parser.add_argument('--base', type=str, default='google', help='base embedding: google, glove, fasttext')
parser.add_argument('--aff', type=str, default='glove', help='affinity info: glove...')
parser.add_argument('--num_trials', type=int, default=1, help='how many different seeds to run with')
parser.add_argument('--model', type=str, default='SGC',
                    choices=['MLP', 'SGC'])
parser.add_argument('--num_folds', type=int, default=5)
parser.add_argument('--n_trials', type=int, default=2, help='number of trials for optuna')
parser.add_argument('--num_train', type=int, default=500)

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

def train(model, optimizer, epoch, idx_train, features, labels, batch_sz=32):
    model.train()
    mse_loss = nn.MSELoss(reduction = "mean")
    permutation = torch.randperm(idx_train.size()[0])
    loss = 0
    for i in range(0, idx_train.size()[0], batch_sz):
        indices = idx_train[permutation[i: (i + batch_sz)]]
        output = model(features, indices)
        optimizer.zero_grad()
        loss_train = mse_loss(output, labels[indices])
        loss_train.backward()
        optimizer.step()
        loss += loss_train.item()
    
    train_loss = loss/(len(idx_train)*labels.shape[1])
    
    return train_loss

def test(model, idx_test, features, labels):
    model.eval()
    with torch.no_grad():
        mse_loss = nn.MSELoss(reduction = "mean")
        output = model(features, idx_test)
        test_loss = mse_loss(output, labels[idx_test]).item()
    
    return test_loss
        
def show_params(args):
    print("hidden: ", args.hidden)
    print("epochs: ", args.epochs)
    print("p: ", args.p)
    print("alpha: ", args.alpha)
    print("base: ", args.base)
    print("delta: ", args.delta)

    
def run_training(fold, params):
    
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    # Build a word graph using the affinity matrix
    adj, features, labels, idx_test, folds_dict = load_data(
        aff=args.aff,
        semantic=args.base,
        delta=args.delta,
        num_folds=args.num_folds,
        num_train=args.num_train,
        seed=args.seed)
    
    features = torch.FloatTensor(features)
    labels = torch.FloatTensor(labels)
    #adj = sparse_mx_to_torch_sparse_tensor(adj)

    idx_test = torch.LongTensor(idx_test)
    idx_train = torch.LongTensor(folds_dict[fold][0])
    idx_valid = torch.LongTensor(folds_dict[fold][1])
    
    propagator = PPRPowerIteration(adj, params['alpha'], params['p'])

    # Set up the GCN Model and the optimizer
    model = GCN(nfeatures=features.shape[1],
                hiddenunits=[args.hidden],
                nout=labels.shape[1],
                drop_prob=params['dropout'],
                propagation=propagator)
    optimizer = optim.Adam(model.parameters(), lr=params['lr'], weight_decay=params['wd'])

    best_loss = np.inf
    early_stop_tolerance = 20
    cnt = 0
    
    if args.cuda:
        model.cuda()
        features = features.cuda()
        labels = labels.cuda()
        #adj = adj.cuda()
        idx_train = idx_train.cuda()

    for epoch in range(args.epochs):
        train(model, optimizer, epoch, idx_train, features, labels, args.batch_size)
        
        train_loss = train(model, optimizer, epoch, idx_train, features, labels, args.batch_size)
        valid_loss = test(model, idx_test=idx_valid, features=features, labels=labels)
        
        if valid_loss < best_loss:
            torch.save(model.state_dict(), 'best_model.bin')
            best_loss = valid_loss
            cnt = 0
        else:
            cnt += 1
        
        if cnt > early_stop_tolerance:
            break 
       
    model.load_state_dict(torch.load('best_model.bin'))
    test_loss = test(model, idx_test=idx_test, features=features, labels=labels)
        
    return valid_loss, test_loss

def objective(trial):
    
    if args.model == 'MLP':
        params={'lr':3e-4,
                'wd':trial.suggest_loguniform("wd", 1e-7, 1e-4),
                'dropout':trial.suggest_float("dropout", 0.0, 0.7, step=0.1),
                'alpha': 0,
                'p': 0
               }
    elif args.model == 'SGC':
        params={'lr':3e-4,
                'wd':trial.suggest_loguniform("wd", 1e-7, 1e-4),
                'dropout':trial.suggest_float("dropout", 0.0, 0.7, step=0.1),
                'alpha': 0,
                'p': 2
               }
        
    valid_losses = []
    test_losses = []
    
    for fold in range(args.num_folds):
        valid_loss, test_loss = run_training(fold, params)
        valid_losses.append(valid_loss)
        test_losses.append(test_loss)
    
    objective_value = np.mean(valid_losses) 
    
    print('TEST RESULTS FOR {}, {}_{} TRIAL {} | Avg. MSE: {:.4f}'.format(args.model, args.base, args.aff, trial.number, np.mean(test_losses)))
    
    return objective_value

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