# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import argparse
import collections
import json
import os
import random
import sys
import time
import uuid
from itertools import chain

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
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import pandas as pd

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
    #############################
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Domain generalization')
    parser.add_argument('--algorithm', type=str, default="ERM")
    parser.add_argument('--task', type=str, default="domain_generalization",
        help='domain_generalization | domain_adaptation')
    parser.add_argument('--hparams', type=str, default={},
        help='JSON-serialized hparams dict')
    parser.add_argument('--hparams_seed', type=int, default=0,
        help='Seed for random hparams (0 means "default hparams")')
    parser.add_argument('--trial_seed', type=int, default=0,
        help='Trial number (used for seeding split_dataset and '
        'random_hparams).')
    parser.add_argument('--seed', type=int, default=5,
        help='Seed for everything else')
    parser.add_argument('--steps', type=int, default=None,
        help='Number of steps. Default is dataset-dependent.')
    parser.add_argument('--checkpoint_freq', type=int, default=None,
        help='Checkpoint every N steps. Default is dataset-dependent.')
    parser.add_argument('--test_envs', type=int, nargs='+', default=[0])
    parser.add_argument('--output_dir', type=str, default="train_output")
    parser.add_argument('--holdout_fraction', type=float, default=0.2)
    parser.add_argument('--uda_holdout_fraction', type=float, default=0)
    parser.add_argument('--skip_model_save', default=0,action='store_true')
    parser.add_argument('--save_model_every_checkpoint', action='store_true')
    
    parser.add_argument('--datapath', type=str, default='../data/')
    parser.add_argument('--dataset', type=str, default='airbnb', choices=['ASSISTments','Hospital_Readmission','Sepsis',
                                                                               'airbnb','channel','jigsaw','wine'])
    parser.add_argument('--mask_option', type=str, default='Gaussian', choices=['Gaussian', 'Uniform', 'Mask'])
    parser.add_argument('--model', type=str, default='mlp', choices=['mlp', 'FTTrans'])
    parser.add_argument('--batch_size', type=int, default=1024)

    
    args = parser.parse_args()
    dataset_path = os.path.join(args.datapath, args.dataset + '/')
    with open(os.path.join(dataset_path, 'info.json')) as f:
        dataset_info = json.load(f)

    # If we ever want to implement checkpointing, just persist these values
    # every once in a while, and then load them from disk here.
    start_step = 0
    algorithm_dict = None

    os.makedirs(args.output_dir, exist_ok=True)
    sys.stdout = misc.Tee(os.path.join(args.output_dir, 'out.txt'))
    sys.stderr = misc.Tee(os.path.join(args.output_dir, 'err.txt'))

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
        hparams = hparams_registry.default_hparams(args.algorithm, 'RotatedMNIST')
    else:
        hparams = hparams_registry.random_hparams(args.algorithm, 'RotatedMNIST',
            misc.seed_hash(args.hparams_seed, args.trial_seed))

    print('HParams:')
    for k, v in sorted(hparams.items()):
        print('\t{}: {}'.format(k, v))

    hparams['mlp_width'] = 256
    hparams['mlp_dropout'] = 0.5
    hparams['mlp_depth'] = 4
    hparams['batch_size'] = args.batch_size
    hparams['model'] = args.model
    hparams['dataset_info'] = dataset_info

    
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    dataset = MulDataset(args, dataset_path, dataset_info)

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

    train_loaders = [InfiniteDataLoader(
        dataset=env,
        weights=env_weights,
        batch_size=hparams['batch_size'],
        num_workers=dataset.N_WORKERS)
        for i, (env, env_weights) in enumerate(in_splits)
        if i not in args.test_envs]
    
    uda_loaders = [InfiniteDataLoader(
        dataset=env,
        weights=env_weights,
        batch_size=hparams['batch_size'],
        num_workers=dataset.N_WORKERS)
        for i, (env, env_weights) in enumerate(uda_splits)
        if i in args.test_envs]

    eval_loaders = [FastDataLoader(
        dataset=env,
        batch_size=64,
        num_workers=dataset.N_WORKERS)
        for env, _ in (in_splits + out_splits + uda_splits)]
    eval_weights = [None for _, weights in (in_splits + out_splits + uda_splits)]
    eval_loader_names = ['env{}_in'.format(i)
        for i in range(len(in_splits))]
    eval_loader_names += ['env{}_out'.format(i)
        for i in range(len(out_splits))]
    eval_loader_names += ['env{}_uda'.format(i)
        for i in range(len(uda_splits))]

    args.worker = 4
    train_bs = args.batch_size
    
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

    steps_per_epoch = min([len(env)/hparams['batch_size'] for env,_ in in_splits])

    n_steps = args.steps or dataset.N_STEPS
    checkpoint_freq = args.checkpoint_freq or dataset.CHECKPOINT_FREQ

    def save_checkpoint(filename):
        if args.skip_model_save:
            return
        save_dict = {
            "args": vars(args),
            "model_input_shape": dataset.input_shape,
            "model_num_classes": dataset.num_classes,
            "model_num_domains": len(dataset) - len(args.test_envs),
            "model_hparams": hparams,
            "model_dict": algorithm.cpu().state_dict()
        }
        torch.save(save_dict, os.path.join(args.output_dir, filename))


    last_results_keys = None
    for step in range(start_step, n_steps):
        step_start_time = time.time()
        minibatches_device = [(x.to(device), y.to(device))
            for x,y in next(train_minibatches_iterator)]
        if args.task == "domain_adaptation":
            uda_device = [x.to(device)
                for x,_ in next(uda_minibatches_iterator)]
        else:
            uda_device = None
        step_vals = algorithm.update(minibatches_device, uda_device)
        checkpoint_vals['step_time'].append(time.time() - step_start_time)

        for key, val in step_vals.items():
            checkpoint_vals[key].append(val)

        if (step % checkpoint_freq == 0) or (step == n_steps - 1):
            results = {
                'step': step,
                'epoch': step / steps_per_epoch,
            }

            for key, val in checkpoint_vals.items():
                results[key] = np.mean(val)

            evals = zip(eval_loader_names, eval_loaders, eval_weights)
            for name, loader, weights in evals:
                acc = misc.accuracy(algorithm, loader, weights, device)
                results[name+'_acc'] = acc

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

            epochs_path = os.path.join(args.output_dir, 'results.jsonl')
            with open(epochs_path, 'a') as f:
                f.write(json.dumps(results, sort_keys=True) + "\n")

            algorithm_dict = algorithm.state_dict()
            start_step = step + 1
            checkpoint_vals = collections.defaultdict(lambda: [])

            records = []
            with open(epochs_path, 'r') as f:
                for line in f:
                    records.append(json.loads(line[:-1]))
            records = Q(records)
            scores = records.map(model_selection.IIDAccuracySelectionMethod._step_acc)
            if scores[-1] == scores.argmax('val_acc'):
                save_checkpoint('IID_best.pkl')
                algorithm.to(device)
            
            if args.save_model_every_checkpoint:
                save_checkpoint(f'model_step{step}.pkl')          
    save_checkpoint('model.pkl')

    with open(os.path.join(args.output_dir, 'done'), 'w') as f:
        f.write('done')
