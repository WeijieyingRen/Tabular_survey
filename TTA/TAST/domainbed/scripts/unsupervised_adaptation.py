# The code is modified from domainbed.scripts.train

import argparse
from argparse import Namespace
import collections
import json
import os
import random
import sys
import time
import uuid
from itertools import chain
import itertools
import copy

import numpy as np
import PIL
import torch
import torchvision
import torch.utils.data

from domainbed import datasets
from domainbed import hparams_registry
from domainbed import algorithms
from domainbed.lib import misc
from domainbed.lib.fast_data_loader import InfiniteDataLoader, FastDataLoader, DataParallelPassthrough
from domainbed import model_selection
from domainbed.lib.query import Q
from domainbed import adapt_algorithms
import itertools
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import pandas as pd
from sklearn import metrics

def read_info(info_path):
    with open(info_path) as f:
        f_list = []
        for line in f:
            tokens = line.strip().split()
            f_list.append(tokens)
    return f_list[:-1]


def read_csv(data_path, shuffle=False):
    X_df= pd.read_csv(data_path,encoding = "ISO-8859-1",delimiter=',')
    return X_df

def Gaussian_to_test(df):
    np.random.seed(42)
    float_columns = df.select_dtypes(include='float')
    noisy_df = float_columns + np.random.normal(0, 1, size=float_columns.shape)
    int_columns = df.select_dtypes(include='int')
    data = pd.concat([noisy_df, int_columns], axis=1)

    return data

def Uniform_to_test(df):
    np.random.seed(42)
    float_columns = df.select_dtypes(include='float')
    noisy_df = float_columns + np.random.uniform(0, 1, size=float_columns.shape)
    int_columns = df.select_dtypes(include='int')
    data = pd.concat([noisy_df, int_columns], axis=1)

    return data

def random_mask_to_test(df):
    np.random.seed(42)
    random_mask = np.random.choice([True, False], size=df.shape, p=[0.9,0.1])
    masked_df = np.where(random_mask, 0, df)
    df = pd.DataFrame(masked_df)
    return df

def read_data_sepsis_train(dataset,y_train_dataset,DATA_DIR):
    train_data_path = os.path.join(DATA_DIR, dataset + '.csv')
    y_data_path = os.path.join(DATA_DIR, y_train_dataset + '.csv')

    X_df = read_csv(train_data_path, shuffle=True)

    y_df = read_csv(y_data_path, shuffle=True)

    data = pd.merge(X_df,y_df)
    class_0_sample = data[data['SepsisLabel'] == 0].sample(frac=0.012)
    class_1_sample = data[data['SepsisLabel'] == 1]
    X = class_0_sample.append(class_1_sample).sample(frac=1) 
    X_train = X.iloc[:,:-1]
    y_df = X.iloc[:,-1]

    X_train = X_train.to_numpy()
    y_train = y_df.to_numpy()
    return X_train,y_train

def read_data(dataset,y_train_dataset,DATA_DIR):
    train_data_path = os.path.join(DATA_DIR, dataset + '.csv')
    y_data_path = os.path.join(DATA_DIR, y_train_dataset + '.csv')

    X_df = read_csv(train_data_path, shuffle=True)
    y_df = read_csv(y_data_path, shuffle=True)
    X_train = X_df.to_numpy()[:,1:]
    y_train = y_df.to_numpy()
    return X_train,y_train

def read_data_cor(dataset,DATA_DIR):
    train_data_path = os.path.join(DATA_DIR, dataset + '.csv')
    data = read_csv(train_data_path, shuffle=True)
    return data

class myDataset(Dataset):
    def __init__(self, data, label):
        self.data = data 
        self.label = label
    
    def __getitem__(self,index):
        return self.data[index], self.label[index]

    def __len__(self):
        return len(self.data)

def data_load(args, dataset_path): 
    ########### normal ##########
    if args.dataset in ['ASSISTments','Hospital_Readmission','Sepsis']:
        train_dataset = 'x_train'
        label_train_dataset = 'y_train'

        test_dataset = 'x_test'
        label_test_dataset = 'y_test'

        ood_test_dataset = 'x_ood_test'
        ood_label_test_dataset = 'y_ood_test'

        X_train,y_train = read_data(train_dataset, label_train_dataset,dataset_path)
        X_test,y_test = read_data(test_dataset,label_test_dataset,dataset_path)

        X_test_ood,y_test_ood = read_data(ood_test_dataset,ood_label_test_dataset,dataset_path)

        y_train = y_train[:,1]
        y_test = y_test[:,1]
        y_test_ood = y_test_ood[:,1]
    ########### corrupt #########
    elif args.dataset in ['airbnb','channel','jigsaw','wine']:
        train_dataset = 'x_train'
        train_label_dataset = 'y_train'

        test_dataset = 'x_test'
        test_label_dataset = 'y_test'

        X_train = read_data_cor(train_dataset,dataset_path)
        y_train = read_data_cor(train_label_dataset,dataset_path)

        X_test = read_data_cor(test_dataset,dataset_path)
        y_test = read_data_cor(test_label_dataset,dataset_path)

        if args.mask_option == 'Gaussian':
            X_test_ood = Gaussian_to_test(X_test)
        elif args.mask_option == 'Uniform':
            X_test_ood = Uniform_to_test(X_test)
        else:
            X_test_ood = random_mask_to_test(X_test)

        y_test_ood = y_test

        X_train = X_train.to_numpy()
        y_train = y_train.to_numpy().squeeze(1)
            
        X_test = X_test.to_numpy()
        y_test = y_test.to_numpy().squeeze(1)

        X_test_ood = X_test_ood.to_numpy()
        y_test_ood = y_test_ood.to_numpy().squeeze(1)
            
        X_train = np.nan_to_num(X_train, nan=0)
        X_test = np.nan_to_num(X_test, nan=0)
        X_test_ood = np.nan_to_num(X_test_ood, nan=0)
    #############################

    print (X_train.shape)
    print (X_test.shape)
    print (X_test_ood.shape)

    data_train = myDataset(torch.from_numpy(X_train).to(torch.float32), torch.from_numpy(y_train).to(torch.int64))
    data_test = myDataset(torch.from_numpy(X_test).to(torch.float32), torch.from_numpy(y_test).to(torch.int64))
    data_test_ood = myDataset(torch.from_numpy(X_test_ood).to(torch.float32), torch.from_numpy(y_test_ood).to(torch.int64))

    return data_train, data_test, data_test_ood

class MultipleDomainDataset:
    N_STEPS = 5001           # Default, subclasses may override
    CHECKPOINT_FREQ = 100    # Default, subclasses may override
    N_WORKERS = 4            # Default, subclasses may override
    ENVIRONMENTS = None      # Subclasses should override
    INPUT_SHAPE = None       # Subclasses should override

    def __getitem__(self, index):
        return self.datasets[index]

    def __len__(self):
        return len(self.datasets)

class MulDataset(MultipleDomainDataset):
    def __init__(self, args, dataset_path, dataset_info):
        super().__init__()
        self.input_shape = dataset_info['n_num_features']+dataset_info['n_cat_features']
        self.num_classes = dataset_info['class_num']
        data_train, data_test, data_test_ood = data_load(args, dataset_path)
        self.datasets = [data_train, data_test, data_test_ood]


class Dataset:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]
    

def generate_featurelized_loader(loader, network, classifier, batch_size=32):
    """
    The classifier adaptation does not need to repeat the heavy forward path, 
    We speeded up the experiments by converting the observations into representations. 
    """
    z_list = []
    y_list = []
    p_list = []
    network.eval()
    classifier.eval()
    for x, y in loader:
        x = x.to(device)
        z = network(x)
        p = classifier(z)
        
        z_list.append(z.detach().cpu())
        y_list.append(y.detach().cpu())
        p_list.append(p.detach().cpu())

    network.train()
    classifier.train()
    z = torch.cat(z_list)
    y = torch.cat(y_list)
    p = torch.cat(p_list)
    ent = softmax_entropy(p)
    py = p.argmax(1).float().cpu().detach()
    dataset1, dataset2 = Dataset(z, y), Dataset(z, py)
    loader1 = torch.utils.data.DataLoader(dataset1, batch_size=batch_size, shuffle=False, drop_last=True)
    loader2 = torch.utils.data.DataLoader(dataset2, batch_size=batch_size, shuffle=False, drop_last=True)
    return loader1, loader2, ent


def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
    """Entropy of softmax distribution from logits."""
    return -(x.softmax(1) * x.log_softmax(1)).sum(1)


def accuracy_ent(network, loader, weights, device, dataset, adapt=False):
    correct = 0
    total = 0
    weights_offset = 0
    ent = 0

    network.eval()
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            if adapt is None:
                p = network(x)
            else:
                p = network(x, adapt)
            if weights is None:
                batch_weights = torch.ones(len(x))
            else:
                batch_weights = weights[weights_offset: weights_offset + len(x)]
                weights_offset += len(x)
            batch_weights = batch_weights.to(device)
            if p.size(1) == 1:
                correct += (p.gt(0).eq(y).float() * batch_weights.view(-1, 1)).sum().item()
            else:
                correct += (p.argmax(1).eq(y).float() * batch_weights).sum().item()
            total += batch_weights.sum().item()
            ent += softmax_entropy(p).sum().item()
        batch = 1024
        for i in range(int(len(dataset.data)/batch)):
            if i == 0:
                p = network(dataset.data[i*batch:(i+1)*batch].cuda()).cpu()
            else:
                p = torch.cat([p,network(dataset.data[i*batch:(i+1)*batch].cuda()).cpu()],dim=0)
        p = torch.cat([p,network(dataset.data[int(len(dataset.data)/batch)*batch:].cuda()).cpu()],dim=0)
        pred = p.argmax(1)
        accuracy = metrics.accuracy_score(dataset.label, pred)          
        f1_score = metrics.f1_score(dataset.label, pred, average='macro')

        print('accuracy:', accuracy)
        print('f1_score:', f1_score)

    network.train()

    return correct / total, ent / total


def accuracy_ent_old(network, loader, weights, device, adapt=False):
    correct = 0
    total = 0
    weights_offset = 0
    ent = 0

    network.eval()
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            if adapt is None:
                p = network(x)
            else:
                p = network(x, adapt)
            if weights is None:
                batch_weights = torch.ones(len(x))
            else:
                batch_weights = weights[weights_offset: weights_offset + len(x)]
                weights_offset += len(x)
            batch_weights = batch_weights.to(device)
            if p.size(1) == 1:
                correct += (p.gt(0).eq(y).float() * batch_weights.view(-1, 1)).sum().item()
            else:
                correct += (p.argmax(1).eq(y).float() * batch_weights).sum().item()
            total += batch_weights.sum().item()
            ent += softmax_entropy(p).sum().item()
    
    network.train()

    return correct / total, ent / total

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Domain generalization')
    parser.add_argument('--input_dir', type=str,default='./train_output/')
    parser.add_argument('--adapt_algorithm', type=str, default="TAST")
    
    parser.add_argument('--datapath', type=str, default='../data/')
    parser.add_argument('--dataset', type=str, default='airbnb', choices=['ASSISTments','Hospital_Readmission','Sepsis',
                                                                               'airbnb','channel','jigsaw','wine'])
    parser.add_argument('--mask_option', type=str, default='Gaussian', choices=['Gaussian', 'Uniform', 'Mask'])
    parser.add_argument('--model', type=str, default='mlp', choices=['mlp', 'FTTrans'])
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--seed', type=int, default=0)
    
    args_in = parser.parse_args()
    dataset_path = os.path.join(args_in.datapath, args_in.dataset + '/')
    with open(os.path.join(dataset_path, 'info.json')) as f:
        dataset_info = json.load(f)
    print(dataset_info)

    epochs_path = os.path.join(args_in.input_dir, 'results.jsonl')
    records = []
    with open(epochs_path, 'r') as f:
        for line in f:
            records.append(json.loads(line[:-1]))
    records = Q(records)
    r = records[0]
    args = Namespace(**r['args'])
    args.input_dir = args_in.input_dir

    if '-' in args_in.adapt_algorithm:
        args.adapt_algorithm, test_batch_size = args_in.adapt_algorithm.split('-')
        args.test_batch_size = int(test_batch_size)
    else:
        args.adapt_algorithm = args_in.adapt_algorithm
        args.test_batch_size = 32  # default

    args.output_dir = args.input_dir
    
    alg_name = args_in.adapt_algorithm

    if args.adapt_algorithm in['T3A', 'TentPreBN', 'TentClf', 'PLClf', 'TAST']:
        use_featurer_cache = False
    else:
        use_featurer_cache = False
    if os.path.exists(os.path.join(args.output_dir, 'done_{}'.format(alg_name))):
        print("{} has already excecuted".format(alg_name))

    # If we ever want to implement checkpointing, just persist these values
    # every once in a while, and then load them from disk here.
    algorithm_dict = None

    sys.stdout = misc.Tee(os.path.join(args.output_dir, 'out_{}.txt'.format(alg_name)))
    sys.stderr = misc.Tee(os.path.join(args.output_dir, 'err_{}.txt'.format(alg_name)))

    print("Environment:")
    print("\tPython: {}".format(sys.version.split(" ")[0]))
    print("\tPyTorch: {}".format(torch.__version__))
    print("\tTorchvision: {}".format(torchvision.__version__))
    print("\tCUDA: {}".format(torch.version.cuda))
    print("\tCUDNN: {}".format(torch.backends.cudnn.version()))
    print("\tNumPy: {}".format(np.__version__))
    print("\tPIL: {}".format(PIL.__version__))

    print('Args:')
    for k, v in sorted(vars(args).items()):
        print('\t{}: {}'.format(k, v))

    if args.hparams_seed == 0:
        hparams = hparams_registry.default_hparams(args.algorithm, args.dataset)
    else:
        hparams = hparams_registry.random_hparams(args.algorithm, args.dataset,
            misc.seed_hash(args.hparams_seed, args.trial_seed))

    print('HParams:')
    for k, v in sorted(hparams.items()):
        print('\t{}: {}'.format(k, v))

    assert os.path.exists(os.path.join(args.output_dir, 'done'))
    assert os.path.exists(os.path.join(args.output_dir, 'IID_best.pkl'))  # IID_best is produced by train.py

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    hparams['mlp_width'] = 256
    hparams['mlp_dropout'] = 0.5
    hparams['mlp_depth'] = 4
    hparams['batch_size'] = args.batch_size
    hparams['model'] = args_in.model
    hparams['dataset_info'] = dataset_info

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    dataset = MulDataset(args_in, dataset_path, dataset_info)

    # Split each env into an 'in-split' and an 'out-split'. We'll train on
    # each in-split except the test envs, and evaluate on all splits.
    
    # To allow unsupervised domain adaptation experiments, we split each test
    # env into 'in-split', 'uda-split' and 'out-split'. The 'in-split' is used
    # by collect_results.py to compute classification accuracies.  The
    # 'out-split' is used by the Oracle model selectino method. The unlabeled
    # samples in 'uda-split' are passed to the algorithm at training time if
    # args.task == "domain_adaptation". If we are interested in comparing
    # domain generalization and domain adaptation results, then domain
    # generalization algorithms should create the same 'uda-splits', which will
    # be discared at training.
    in_splits = []
    out_splits = []
    uda_splits = []
    for env_i, env in enumerate(dataset):
        uda = []
        out, in_ = misc.split_dataset(env,
            int(len(env)*args.holdout_fraction),
            misc.seed_hash(args.trial_seed, env_i))

        if env_i in args.test_envs:
            uda, in_ = misc.split_dataset(in_,
                int(len(in_)*args.uda_holdout_fraction),
                misc.seed_hash(args.trial_seed, env_i))

        if hparams['class_balanced']:
            in_weights = misc.make_weights_for_balanced_classes(in_)
            out_weights = misc.make_weights_for_balanced_classes(out)
            if uda is not None:
                uda_weights = misc.make_weights_for_balanced_classes(uda)
        else:
            in_weights, out_weights, uda_weights = None, None, None
        in_splits.append((in_, in_weights))
        out_splits.append((out, out_weights))
        if len(uda):
            uda_splits.append((uda, uda_weights))

    # Use out splits as training data (to fair comparison with train.py)
    train_loaders = [FastDataLoader(
        dataset=env,
        batch_size=hparams['batch_size'],
        num_workers=dataset.N_WORKERS)
        for i, (env, env_weights) in enumerate(out_splits)
        if i in args.test_envs]
    
    uda_loaders = [InfiniteDataLoader(
        dataset=env,
        weights=env_weights,
        batch_size=hparams['batch_size'],
        num_workers=dataset.N_WORKERS)
        for i, (env, env_weights) in enumerate(uda_splits)
        if i in args.test_envs]

    eval_loaders = [FastDataLoader(
        dataset=env,
        batch_size=args.test_batch_size,
        num_workers=dataset.N_WORKERS)
        for env, _ in (in_splits + out_splits + uda_splits)]
    eval_weights = [None for _, weights in (in_splits + out_splits + uda_splits)]
    eval_loader_names = ['env{}_in'.format(i)
        for i in range(len(in_splits))]
    eval_loader_names += ['env{}_out'.format(i)
        for i in range(len(out_splits))]
    eval_loader_names += ['env{}_uda'.format(i)
        for i in range(len(uda_splits))]

    algorithm_class = algorithms.get_algorithm_class(args.algorithm)
    algorithm = algorithm_class(dataset.input_shape, dataset.num_classes,
        len(dataset) - len(args.test_envs), hparams)

    if algorithm_dict is not None:
        algorithm.load_state_dict(algorithm_dict)

    algorithm.to(device)
    if hasattr(algorithm, 'network'):
        algorithm.network = DataParallelPassthrough(algorithm.network)
    else:
        for m in algorithm.children():
            m = DataParallelPassthrough(m)

    train_minibatches_iterator = zip(*train_loaders)
    uda_minibatches_iterator = zip(*uda_loaders)
    checkpoint_vals = collections.defaultdict(lambda: [])

    # load trained model
    ckpt = torch.load(os.path.join(args.output_dir, 'IID_best.pkl'))
    algorithm_dict = ckpt['model_dict']
    if algorithm_dict is not None:
        algorithm.load_state_dict(algorithm_dict)

    # Evaluate base model
    print("Base model's results")
    results = {}
    evals = zip(eval_loader_names, eval_loaders, eval_weights)
    for name, loader, weights in evals:
        acc, ent = accuracy_ent(algorithm, loader, weights, device, dataset[2], adapt=None)
        results[name+'_acc'] = acc
        results[name+'_ent'] = ent
    results_keys = sorted(results.keys())
    misc.print_row(results_keys, colwidth=12)
    misc.print_row([results[key] for key in results_keys], colwidth=12)

    print("\nAfter {}".format(alg_name))
    # Cache the inference results
    if use_featurer_cache:
        original_evals = zip(eval_loader_names, eval_loaders, eval_weights)
        loaders = []
        for name, loader, weights in original_evals:
            loader1, loader2, ent = generate_featurelized_loader(loader, network=algorithm.featurizer, classifier=algorithm.classifier, batch_size=32)
            loaders.append((name, loader1, weights))
    else:
        loaders = zip(eval_loader_names, eval_loaders, eval_weights)
    
    evals = []
    for name, loader, weights in loaders:
        if name in ['env{}_in'.format(i) for i in args.test_envs]:
            train_loader = (name, loader, weights)
        else:
            evals.append((name, loader, weights))

    last_results_keys = None
    adapt_algorithm_class = adapt_algorithms.get_algorithm_class(
        args.adapt_algorithm)
    
    if args.adapt_algorithm in ['T3A']:
        adapt_hparams_dict = {
            'filter_K': [1, 5, 20, 50, 100, -1], 
        }
    elif args.adapt_algorithm in ['TentFull', 'TentPreBN', 'TentClf', 'TentNorm']:
        adapt_hparams_dict = {
            'alpha': [0.1, 1.0, 10.0],
            'gamma': [1, 3]
        }
    elif args.adapt_algorithm in ['PseudoLabel', 'PLClf']:
        adapt_hparams_dict = {
            'alpha': [0.1, 1.0, 10.0],
            'gamma': [1, 3], 
            'beta': [0.9]
        }
    elif args.adapt_algorithm in ['SHOT', 'SHOTIM']:
        adapt_hparams_dict = {
            'alpha': [0.1, 1.0, 10.0],
            'gamma': [1, 3], 
            'beta': [0.9], 
            'theta': [0.1], 
        }
    elif args.adapt_algorithm in ['TAST_BN']:
        adapt_hparams_dict = {
            'filter_K': [1, 5, 20],
            'gamma': [1, 3],
            'lr': [1e-3],
            'tau': [10],
            'k': [1, 2, 4, 8],
        }
    elif args.adapt_algorithm in ['TAST']:
        adapt_hparams_dict = {
            'num_ensemble': [1, 5, 10, 20],
            'filter_K': [1, 5, 20, 50, 100, -1],
            'gamma': [1, 3],
            'lr': [1e-3],
            'tau': [10],
            'k': [1, 2, 4, 8],
            'init_mode': ['kaiming_normal']
        }
    else:
        raise Exception("Not Implemented Error")
    product = [x for x in itertools.product(*adapt_hparams_dict.values())]
    adapt_hparams_list = [dict(zip(adapt_hparams_dict.keys(), r)) for r in product]

    for adapt_hparams in adapt_hparams_list:
        adapt_hparams['cached_loader'] = use_featurer_cache
        adapted_algorithm = adapt_algorithm_class(dataset.input_shape, dataset.num_classes,
            len(dataset) - len(args.test_envs), adapt_hparams, algorithm
        )
        # adapted_algorithm = DataParallelPassthrough(adapted_algorithm)
        adapted_algorithm.to(device)
        
        results = adapt_hparams

        for key, val in checkpoint_vals.items():
            results[key] = np.mean(val)

        # ## Usual evaluation
        for name, loader, weights in evals:
            acc, ent = accuracy_ent_old(adapted_algorithm, loader, weights, device, adapt=True)
            results[name+'_acc'] = acc
            results[name+'_ent'] = ent
            adapted_algorithm.reset()

        name, loader, weights = train_loader
        acc, ent = accuracy_ent(adapted_algorithm, loader, weights, device, dataset[2], adapt=True)
        results[name+'_acc'] = acc
        results[name+'_ent'] = ent

        del adapt_hparams['cached_loader']
        results_keys = sorted(results.keys())

        if results_keys != last_results_keys:
            misc.print_row(results_keys, colwidth=12)
            last_results_keys = results_keys
        misc.print_row([results[key] for key in results_keys],
            colwidth=12)

        results.update({
            'hparams': hparams,
            'args': vars(args)    
        })
        # save file
        epochs_path = os.path.join(args.output_dir, 'results_{}.jsonl'.format(alg_name))
        with open(epochs_path, 'a') as f:
            f.write(json.dumps(results, sort_keys=True) + "\n")

    # create done file
    with open(os.path.join(args.output_dir, 'done_{}'.format(alg_name)), 'w') as f:
        f.write('done')

        
