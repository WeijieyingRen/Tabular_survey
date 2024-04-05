import argparse
import os, sys
import os.path as osp
import torchvision
import numpy as np
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
import network, loss
from torch.utils.data import DataLoader
from data_list import ImageList, ImageList_idx
import random, pdb, math, copy
from tqdm import tqdm
from scipy.spatial.distance import cdist
from sklearn.metrics import confusion_matrix
from torch.utils.data import Dataset
import pandas as pd
from sklearn import metrics
from tab_transformer_pytorch import FTTransformer

def op_copy(optimizer):
    for param_group in optimizer.param_groups:
        param_group['lr0'] = param_group['lr']
    return optimizer

def lr_scheduler(optimizer, iter_num, max_iter, gamma=10, power=0.75):
    decay = (1 + gamma * iter_num / max_iter) ** (-power)
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr0'] * decay
        param_group['weight_decay'] = 1e-3
        param_group['momentum'] = 0.9
        param_group['nesterov'] = True
    return optimizer

def image_train(resize_size=256, crop_size=224, alexnet=False):
  if not alexnet:
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
  else:
    normalize = Normalize(meanfile='./ilsvrc_2012_mean.npy')
  return  transforms.Compose([
        transforms.Resize((resize_size, resize_size)),
        transforms.RandomCrop(crop_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ])

def image_test(resize_size=256, crop_size=224, alexnet=False):
  if not alexnet:
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
  else:
    normalize = Normalize(meanfile='./ilsvrc_2012_mean.npy')
  return  transforms.Compose([
        transforms.Resize((resize_size, resize_size)),
        transforms.CenterCrop(crop_size),
        transforms.ToTensor(),
        normalize
    ])

def data_load_old(args): 
    ## prepare data
    dsets = {}
    dset_loaders = {}
    train_bs = args.batch_size
    txt_src = open(args.s_dset_path).readlines()
    txt_test = open(args.test_dset_path).readlines()

    if not args.da == 'uda':
        label_map_s = {}
        for i in range(len(args.src_classes)):
            label_map_s[args.src_classes[i]] = i
        
        new_src = []
        for i in range(len(txt_src)):
            rec = txt_src[i]
            reci = rec.strip().split(' ')
            if int(reci[1]) in args.src_classes:
                line = reci[0] + ' ' + str(label_map_s[int(reci[1])]) + '\n'   
                new_src.append(line)
        txt_src = new_src.copy()

        new_tar = []
        for i in range(len(txt_test)):
            rec = txt_test[i]
            reci = rec.strip().split(' ')
            if int(reci[1]) in args.tar_classes:
                if int(reci[1]) in args.src_classes:
                    line = reci[0] + ' ' + str(label_map_s[int(reci[1])]) + '\n'   
                    new_tar.append(line)
                else:
                    line = reci[0] + ' ' + str(len(label_map_s)) + '\n'   
                    new_tar.append(line)
        txt_test = new_tar.copy()

    if args.trte == "val":
        dsize = len(txt_src)
        tr_size = int(0.9*dsize)
        # print(dsize, tr_size, dsize - tr_size)
        tr_txt, te_txt = torch.utils.data.random_split(txt_src, [tr_size, dsize - tr_size])
    else:
        dsize = len(txt_src)
        tr_size = int(0.9*dsize)
        _, te_txt = torch.utils.data.random_split(txt_src, [tr_size, dsize - tr_size])
        tr_txt = txt_src

    dsets["source_tr"] = ImageList(tr_txt, transform=image_train())
    dset_loaders["source_tr"] = DataLoader(dsets["source_tr"], batch_size=train_bs, shuffle=True, num_workers=args.worker, drop_last=False)
    dsets["source_te"] = ImageList(te_txt, transform=image_test())
    dset_loaders["source_te"] = DataLoader(dsets["source_te"], batch_size=train_bs, shuffle=True, num_workers=args.worker, drop_last=False)
    dsets["test"] = ImageList(txt_test, transform=image_test())
    dset_loaders["test"] = DataLoader(dsets["test"], batch_size=train_bs*2, shuffle=True, num_workers=args.worker, drop_last=False)

    return dset_loaders

def read_info(info_path):
    with open(info_path) as f:
        f_list = []
        for line in f:
            tokens = line.strip().split()
            f_list.append(tokens)
    return f_list[:-1]

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

def read_csv(data_path, shuffle=False):
    X_df= pd.read_csv(data_path,encoding = "ISO-8859-1",delimiter=',')
    return X_df

def read_data_cor(dataset,DATA_DIR):
    train_data_path = os.path.join(DATA_DIR, dataset + '.csv')
    data = read_csv(train_data_path, shuffle=True)
    return data

def read_data(dataset,y_train_dataset,DATA_DIR):
    train_data_path = os.path.join(DATA_DIR, dataset + '.csv')
    y_data_path = os.path.join(DATA_DIR, y_train_dataset + '.csv')

    X_df = read_csv(train_data_path, shuffle=True)
    y_df = read_csv(y_data_path, shuffle=True)
    X_train = X_df.to_numpy()[:,1:]
    y_train = y_df.to_numpy()
    return X_train,y_train

class myDataset(Dataset):
    def __init__(self, data, label):
        self.data = data 
        self.label = label
    
    def __getitem__(self,index):
        return self.data[index], self.label[index]

    def __len__(self):
        return len(self.data)

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

def data_load(args, dataset_path): 
    train_bs = args.batch_size
    dset_loaders = {}

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

    data_test_ood = myDataset(torch.from_numpy(X_test_ood).to(torch.float32), torch.from_numpy(y_test_ood).to(torch.int64))
    
    dset_loaders["target"] = DataLoader(data_test_ood, batch_size=train_bs*2, shuffle=True, num_workers=args.worker, drop_last=False)
    dset_loaders["test"] = DataLoader(data_test_ood, batch_size=train_bs*2, shuffle=True, num_workers=args.worker, drop_last=False)

    return dset_loaders

def cal_acc(loader, netF, netB, netC, flag=False):
    start_test = True
    with torch.no_grad():
        iter_test = iter(loader)
        for i in range(len(loader)):
            data = next(iter_test)
            inputs = data[0]
            labels = data[1]
            inputs = inputs.cuda()
            outputs = netC(netB(netF(inputs)))
            if start_test:
                all_output = outputs.float().cpu()
                all_label = labels.float()
                start_test = False
            else:
                all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)
    _, predict = torch.max(all_output, 1)
    accuracy = metrics.accuracy_score(all_label, predict)          
    f1_score = metrics.f1_score(all_label, predict, average='macro')
    
    print ('ood accuracy is: {}'.format(accuracy))
    print ('ood f1 score is:{}'.format(f1_score))
    
    accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])
    mean_ent = torch.mean(loss.Entropy(nn.Softmax(dim=1)(all_output))).cpu().data.item()

    if flag:
        matrix = confusion_matrix(all_label, torch.squeeze(predict).float())
        acc = matrix.diagonal()/matrix.sum(axis=1) * 100
        aacc = acc.mean()
        aa = [str(np.round(i, 2)) for i in acc]
        acc = ' '.join(aa)
        return aacc, acc
    else:
        return accuracy*100, mean_ent

class Model(torch.nn.Module):
    def __init__(self, input_shape, output_shape):
        super(Model,self).__init__()
        
        self.dense1 = nn.Linear(in_features=input_shape, out_features=48, bias=False)
        self.bn1 = nn.BatchNorm1d(48)
        self.dense2 = nn.Linear(in_features=48, out_features=16, bias=False)
        self.bn2 = nn.BatchNorm1d(16)
        self.dense3 = nn.Linear(in_features=16, out_features=output_shape, bias=False)
        self.relu=torch.nn.ReLU()
        self.sm = nn.Softmax(dim=1)
        self.in_features = output_shape

    def forward(self, inputs):
        x = self.dense1(inputs)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dense2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.dense3(x)
        x = self.sm(x)
        return x

class FT_Model(nn.Module):
    def __init__(self, n_outputs, FTTrans, num_idx, cat_idx):
        super(FT_Model, self).__init__()
        self.tabTrans = FTTrans
        self.in_features = n_outputs
        self.num_idx = num_idx
        self.cat_idx = cat_idx
 
    def forward(self, x):
        if len(self.num_idx) != 0:
            num = x[:,self.num_idx]
        else:
            num = torch.tensor([])
        if len(self.cat_idx) != 0:
            cat = x[:,self.cat_idx].to(torch.long)
        else:
            cat = torch.tensor([]).to(torch.long)
        x = self.tabTrans(cat, num)
        return x

def train_target(args, dataset_path, dataset_info):
    dset_loaders = data_load(args, dataset_path)
    
    if args.model == 'mlp':
        netF = Model(dataset_info['n_num_features']+dataset_info['n_cat_features'], dataset_info['class_num']).cuda()
    elif args.model == 'FTTrans':
        FTTrans = FTTransformer(
            categories = tuple(dataset_info['cat_clsnum']),             # tuple containing the number of unique values within each category
            num_continuous = dataset_info['n_num_features'],            # number of continuous values
            dim = 32,                                                   # dimension, paper set at 32
            dim_out = 48,                                               # binary prediction, but could be anything
            depth = 6,                                                  # depth, paper recommended 6
            heads = 8,                                                  # heads, paper recommends 8
            attn_dropout = 0.1,                                         # post-attention dropout
            ff_dropout = 0.1                                            # feed forward dropout
        )
        netF = FT_Model(48, FTTrans, dataset_info['num_index'], dataset_info['cat_index']).cuda()
    
    netB = network.feat_bottleneck(type=args.classifier, feature_dim=netF.in_features, bottleneck_dim=args.bottleneck).cuda()
    netC = network.feat_classifier(type=args.layer, class_num = args.class_num, bottleneck_dim=args.bottleneck).cuda()

    modelpath = args.output_dir_src + '/source_F.pt'   
    netF.load_state_dict(torch.load(modelpath))
    modelpath = args.output_dir_src + '/source_B.pt'   
    netB.load_state_dict(torch.load(modelpath))
    modelpath = args.output_dir_src + '/source_C.pt'    
    netC.load_state_dict(torch.load(modelpath))
    netC.eval()
    for k, v in netC.named_parameters():
        v.requires_grad = False

    param_group = []
    for k, v in netF.named_parameters():
        if args.lr_decay1 > 0:
            param_group += [{'params': v, 'lr': args.lr * args.lr_decay1}]
        else:
            v.requires_grad = False
    for k, v in netB.named_parameters():
        if args.lr_decay2 > 0:
            param_group += [{'params': v, 'lr': args.lr * args.lr_decay2}]
        else:
            v.requires_grad = False

    optimizer = optim.SGD(param_group)
    optimizer = op_copy(optimizer)

    max_iter = args.max_epoch * len(dset_loaders["target"])
    interval_iter = max_iter // args.interval
    iter_num = 0

    while iter_num < max_iter:
        try:
            inputs_test, _ = next(iter_test)
            i += 1
            tar_idx = list(range(i*128,(i+1)*128))
        except:
            iter_test = iter(dset_loaders["target"])
            i = 0
            tar_idx = list(range(i*128,(i+1)*128))
            inputs_test, _ = next(iter_test)

        if inputs_test.size(0) == 1:
            continue

        if iter_num % interval_iter == 0 and args.cls_par > 0:
            netF.eval()
            netB.eval()
            mem_label = obtain_label(dset_loaders['test'], netF, netB, netC, args)
            mem_label = torch.from_numpy(mem_label).cuda()
            netF.train()
            netB.train()

        inputs_test = inputs_test.cuda()

        iter_num += 1
        lr_scheduler(optimizer, iter_num=iter_num, max_iter=max_iter)

        features_test = netB(netF(inputs_test))
        outputs_test = netC(features_test)

        if args.cls_par > 0:
            tar_idx = tar_idx[:outputs_test.shape[0]]
            pred = mem_label[tar_idx]
            classifier_loss = nn.CrossEntropyLoss()(outputs_test, pred)
            classifier_loss *= args.cls_par
            if iter_num < interval_iter and args.dset == "VISDA-C":
                classifier_loss *= 0
        else:
            classifier_loss = torch.tensor(0.0).cuda()

        if args.ent:
            softmax_out = nn.Softmax(dim=1)(outputs_test)
            entropy_loss = torch.mean(loss.Entropy(softmax_out))
            if args.gent:
                msoftmax = softmax_out.mean(dim=0)
                gentropy_loss = torch.sum(-msoftmax * torch.log(msoftmax + args.epsilon))
                entropy_loss -= gentropy_loss
            im_loss = entropy_loss * args.ent_par
            classifier_loss += im_loss

        optimizer.zero_grad()
        classifier_loss.backward()
        optimizer.step()

        if iter_num % interval_iter == 0 or iter_num == max_iter:
            netF.eval()
            netB.eval()
            if args.dset=='VISDA-C':
                acc_s_te, acc_list = cal_acc(dset_loaders['test'], netF, netB, netC, True)
                log_str = 'Task: {}, Iter:{}/{}; Accuracy = {:.2f}%'.format(args.name, iter_num, max_iter, acc_s_te) + '\n' + acc_list
            else:
                acc_s_te, _ = cal_acc(dset_loaders['test'], netF, netB, netC, False)
                log_str = 'Task: {}, Iter:{}/{}; Accuracy = {:.2f}%'.format(args.name, iter_num, max_iter, acc_s_te)

            args.out_file.write(log_str + '\n')
            args.out_file.flush()
            print(log_str+'\n')
            netF.train()
            netB.train()

    if args.issave:   
        torch.save(netF.state_dict(), osp.join(args.output_dir, "target_F_" + args.savename + ".pt"))
        torch.save(netB.state_dict(), osp.join(args.output_dir, "target_B_" + args.savename + ".pt"))
        torch.save(netC.state_dict(), osp.join(args.output_dir, "target_C_" + args.savename + ".pt"))
        
    return netF, netB, netC

def print_args(args):
    s = "==========================================\n"
    for arg, content in args.__dict__.items():
        s += "{}:{}\n".format(arg, content)
    return s

def obtain_label(loader, netF, netB, netC, args):
    start_test = True
    with torch.no_grad():
        iter_test = iter(loader)
        for _ in range(len(loader)):
            data = next(iter_test)
            inputs = data[0]
            labels = data[1]
            inputs = inputs.cuda()
            feas = netB(netF(inputs))
            outputs = netC(feas)
            if start_test:
                all_fea = feas.float().cpu()
                all_output = outputs.float().cpu()
                all_label = labels.float()
                start_test = False
            else:
                all_fea = torch.cat((all_fea, feas.float().cpu()), 0)
                all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)

    all_output = nn.Softmax(dim=1)(all_output)
    ent = torch.sum(-all_output * torch.log(all_output + args.epsilon), dim=1)
    unknown_weight = 1 - ent / np.log(args.class_num)
    _, predict = torch.max(all_output, 1)

    accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])
    if args.distance == 'cosine':
        all_fea = torch.cat((all_fea, torch.ones(all_fea.size(0), 1)), 1)
        all_fea = (all_fea.t() / torch.norm(all_fea, p=2, dim=1)).t()

    all_fea = all_fea.float().cpu().numpy()
    K = all_output.size(1)
    aff = all_output.float().cpu().numpy()

    for _ in range(2):
        initc = aff.transpose().dot(all_fea)
        initc = initc / (1e-8 + aff.sum(axis=0)[:,None])
        cls_count = np.eye(K)[predict].sum(axis=0)
        labelset = np.where(cls_count>args.threshold)
        labelset = labelset[0]

        dd = cdist(all_fea, initc[labelset], args.distance)
        pred_label = dd.argmin(axis=1)
        predict = labelset[pred_label]

        aff = np.eye(K)[predict]

    acc = np.sum(predict == all_label.float().numpy()) / len(all_fea)
    log_str = 'Accuracy = {:.2f}% -> {:.2f}%'.format(accuracy * 100, acc * 100)

    args.out_file.write(log_str + '\n')
    args.out_file.flush()
    print(log_str+'\n')

    return predict.astype('int')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='SHOT')
    parser.add_argument('--gpu_id', type=str, nargs='?', default='0', help="device id to run")
    parser.add_argument('--s', type=int, default=0, help="source")
    parser.add_argument('--t', type=int, default=1, help="target")
    parser.add_argument('--max_epoch', type=int, default=15, help="max iterations")
    parser.add_argument('--interval', type=int, default=15)
    parser.add_argument('--batch_size', type=int, default=64, help="batch_size")
    parser.add_argument('--worker', type=int, default=4, help="number of workers")
    parser.add_argument('--dset', type=str, default='office-home', choices=['VISDA-C', 'office', 'office-home', 'office-caltech'])
    parser.add_argument('--lr', type=float, default=1e-2, help="learning rate")
    parser.add_argument('--net', type=str, default='resnet50', help="alexnet, vgg16, resnet50, res101")
    parser.add_argument('--seed', type=int, default=2020, help="random seed")
 
    parser.add_argument('--gent', type=bool, default=True)
    parser.add_argument('--ent', type=bool, default=True)
    parser.add_argument('--threshold', type=int, default=0)
    parser.add_argument('--cls_par', type=float, default=0.3)
    parser.add_argument('--ent_par', type=float, default=1.0)
    parser.add_argument('--lr_decay1', type=float, default=0.1)
    parser.add_argument('--lr_decay2', type=float, default=1.0)

    parser.add_argument('--bottleneck', type=int, default=256)
    parser.add_argument('--epsilon', type=float, default=1e-5)
    parser.add_argument('--layer', type=str, default="wn", choices=["linear", "wn"])
    parser.add_argument('--classifier', type=str, default="bn", choices=["ori", "bn"])
    parser.add_argument('--distance', type=str, default='cosine', choices=["euclidean", "cosine"])  
    parser.add_argument('--output', type=str, default='san')
    parser.add_argument('--output_src', type=str, default='san')
    parser.add_argument('--da', type=str, default='uda', choices=['uda', 'pda'])
    parser.add_argument('--issave', type=bool, default=True)
    
    parser.add_argument('--datapath', type=str, default='../data/')
    parser.add_argument('--dataset', type=str, default='airbnb', choices=['ASSISTments','Hospital_Readmission','Sepsis',
                                                                               'airbnb','channel','jigsaw','wine'])
    parser.add_argument('--mask_option', type=str, default='Gaussian', choices=['Gaussian', 'Uniform', 'Mask'])
    parser.add_argument('--model', type=str, default='mlp', choices=['mlp', 'FTTrans'])
    
    args = parser.parse_args()
    dataset_path = os.path.join(args.datapath, args.dataset + '/')
    with open(os.path.join(dataset_path, 'info.json')) as f:
        dataset_info = json.load(f)
    print(dataset_info)

    if args.dset == 'office-home':
        names = ['Art', 'Clipart']
        args.class_num = dataset_info['class_num']
        
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    SEED = args.seed
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    # torch.backends.cudnn.deterministic = True

    for i in range(len(names)):
        if i == args.s:
            continue
        args.t = i

        folder = './data/'
        args.s_dset_path = folder + args.dset + '/' + names[args.s] + '_list.txt'
        args.t_dset_path = folder + args.dset + '/' + names[args.t] + '_list.txt'
        args.test_dset_path = folder + args.dset + '/' + names[args.t] + '_list.txt'

        if args.dset == 'office-home':
            if args.da == 'pda':
                args.class_num = 65
                args.src_classes = [i for i in range(65)]
                args.tar_classes = [i for i in range(25)]

        args.output_dir_src = osp.join(args.output_src, args.da, args.dset, names[args.s][0].upper())
        args.output_dir = osp.join(args.output, args.da, args.dset, names[args.s][0].upper()+names[args.t][0].upper())
        args.name = names[args.s][0].upper()+names[args.t][0].upper()

        if not osp.exists(args.output_dir):
            os.system('mkdir -p ' + args.output_dir)
        if not osp.exists(args.output_dir):
            os.mkdir(args.output_dir)

        args.savename = 'par_' + str(args.cls_par)
        if args.da == 'pda':
            args.gent = ''
            args.savename = 'par_' + str(args.cls_par) + '_thr' + str(args.threshold)
        args.out_file = open(osp.join(args.output_dir, 'log_' + args.savename + '.txt'), 'w')
        args.out_file.write(print_args(args)+'\n')
        args.out_file.flush()
        train_target(args)