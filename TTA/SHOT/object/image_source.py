import argparse
import os, sys
import os.path as osp
import torchvision
import numpy as np
import torch
import json
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
import network, loss
from torch.utils.data import DataLoader
from data_list import ImageList
import random, pdb, math, copy
from tqdm import tqdm
from loss import CrossEntropyLabelSmooth
from scipy.spatial.distance import cdist
from sklearn.metrics import confusion_matrix
from sklearn.cluster import KMeans
from torch.utils.data import Dataset
import pandas as pd
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
    #return f_list[:-1], int(f_list[-1][-1])
    return f_list[:-1]


def read_csv(data_path, shuffle=False):
    #D = pd.read_csv(data_path, header=None)
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

    data_train = myDataset(torch.from_numpy(X_train).to(torch.float32), torch.from_numpy(y_train).to(torch.int64))
    data_test = myDataset(torch.from_numpy(X_test).to(torch.float32), torch.from_numpy(y_test).to(torch.int64))
    data_test_ood = myDataset(torch.from_numpy(X_test_ood).to(torch.float32), torch.from_numpy(y_test_ood).to(torch.int64))

    dset_loaders["source_tr"] = DataLoader(data_train, batch_size=train_bs, shuffle=True, num_workers=args.worker, drop_last=False)
    dset_loaders["source_te"] = DataLoader(data_test, batch_size=train_bs, shuffle=True, num_workers=args.worker, drop_last=False)
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

    all_output = nn.Softmax(dim=1)(all_output)
    _, predict = torch.max(all_output, 1)
    accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])
    mean_ent = torch.mean(loss.Entropy(all_output)).cpu().data.item()
   
    if flag:
        matrix = confusion_matrix(all_label, torch.squeeze(predict).float())
        acc = matrix.diagonal()/matrix.sum(axis=1) * 100
        aacc = acc.mean()
        aa = [str(np.round(i, 2)) for i in acc]
        acc = ' '.join(aa)
        return aacc, acc
    else:
        return accuracy*100, mean_ent

def cal_acc_oda(loader, netF, netB, netC):
    start_test = True
    with torch.no_grad():
        iter_test = iter(loader)
        for i in range(len(loader)):
            data = iter_test.next()
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

    all_output = nn.Softmax(dim=1)(all_output)
    _, predict = torch.max(all_output, 1)
    ent = torch.sum(-all_output * torch.log(all_output + args.epsilon), dim=1) / np.log(args.class_num)
    ent = ent.float().cpu()
    initc = np.array([[0], [1]])
    kmeans = KMeans(n_clusters=2, random_state=0, init=initc, n_init=1).fit(ent.reshape(-1,1))
    threshold = (kmeans.cluster_centers_).mean()

    predict[ent>threshold] = args.class_num
    matrix = confusion_matrix(all_label, torch.squeeze(predict).float())
    matrix = matrix[np.unique(all_label).astype(int),:]

    acc = matrix.diagonal()/matrix.sum(axis=1) * 100
    unknown_acc = acc[-1:].item()

    return np.mean(acc[:-1]), np.mean(acc), unknown_acc
    # return np.mean(acc), np.mean(acc[:-1])

def train_source(args, dataset_path, dataset_info):
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

    param_group = []
    learning_rate = args.lr
    for k, v in netF.named_parameters():
        param_group += [{'params': v, 'lr': learning_rate*0.1}]
    for k, v in netB.named_parameters():
        param_group += [{'params': v, 'lr': learning_rate}]
    for k, v in netC.named_parameters():
        param_group += [{'params': v, 'lr': learning_rate}]   
    optimizer = optim.SGD(param_group)
    optimizer = op_copy(optimizer)

    acc_init = 0
    max_iter = args.max_epoch * len(dset_loaders["source_tr"])
    interval_iter = max_iter // 10
    iter_num = 0

    netF.train()
    netB.train()
    netC.train()

    while iter_num < max_iter:
        try:
            inputs_source, labels_source = next(iter_source)
        except:
            iter_source = iter(dset_loaders["source_tr"])
            inputs_source, labels_source = next(iter_source)

        if inputs_source.size(0) == 1:
            continue

        iter_num += 1
        lr_scheduler(optimizer, iter_num=iter_num, max_iter=max_iter)

        inputs_source, labels_source = inputs_source.cuda(), labels_source.cuda()
        outputs_source = netC(netB(netF(inputs_source)))
        classifier_loss = CrossEntropyLabelSmooth(num_classes=args.class_num, epsilon=args.smooth)(outputs_source, labels_source)            
        
        optimizer.zero_grad()
        classifier_loss.backward()
        optimizer.step()

        if iter_num % interval_iter == 0 or iter_num == max_iter:
            netF.eval()
            netB.eval()
            netC.eval()
            if args.dset=='VISDA-C':
                acc_s_te, acc_list = cal_acc(dset_loaders['source_te'], netF, netB, netC, True)
                log_str = 'Task: {}, Iter:{}/{}; Accuracy = {:.2f}%'.format(args.name_src, iter_num, max_iter, acc_s_te) + '\n' + acc_list
            else:
                acc_s_te, _ = cal_acc(dset_loaders['source_te'], netF, netB, netC, False)
                log_str = 'Task: {}, Iter:{}/{}; Accuracy = {:.2f}%'.format(args.name_src, iter_num, max_iter, acc_s_te)
            args.out_file.write(log_str + '\n')
            args.out_file.flush()
            print(log_str+'\n')

            if acc_s_te >= acc_init:
                acc_init = acc_s_te
                best_netF = netF.state_dict()
                best_netB = netB.state_dict()
                best_netC = netC.state_dict()

            netF.train()
            netB.train()
            netC.train()
                
    torch.save(best_netF, osp.join(args.output_dir_src, "source_F.pt"))
    torch.save(best_netB, osp.join(args.output_dir_src, "source_B.pt"))
    torch.save(best_netC, osp.join(args.output_dir_src, "source_C.pt"))

    return netF, netB, netC

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

def test_target(args, dataset_path, dataset_info):
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
    
    args.modelpath = args.output_dir_src + '/source_F.pt'   
    netF.load_state_dict(torch.load(args.modelpath))
    args.modelpath = args.output_dir_src + '/source_B.pt'   
    netB.load_state_dict(torch.load(args.modelpath))
    args.modelpath = args.output_dir_src + '/source_C.pt'   
    netC.load_state_dict(torch.load(args.modelpath))
    netF.eval()
    netB.eval()
    netC.eval()

    if args.da == 'oda':
        acc_os1, acc_os2, acc_unknown = cal_acc_oda(dset_loaders['test'], netF, netB, netC)
        log_str = '\nTraining: {}, Task: {}, Accuracy = {:.2f}% / {:.2f}% / {:.2f}%'.format(args.trte, args.name, acc_os2, acc_os1, acc_unknown)
    else:
        if args.dset=='VISDA-C':
            acc, acc_list = cal_acc(dset_loaders['test'], netF, netB, netC, True)
            log_str = '\nTraining: {}, Task: {}, Accuracy = {:.2f}%'.format(args.trte, args.name, acc) + '\n' + acc_list
        else:
            acc, _ = cal_acc(dset_loaders['test'], netF, netB, netC, False)
            log_str = '\nTraining: {}, Task: {}, Accuracy = {:.2f}%'.format(args.trte, args.name, acc)

    args.out_file.write(log_str)
    args.out_file.flush()
    print(log_str)

def print_args(args):
    s = "==========================================\n"
    for arg, content in args.__dict__.items():
        s += "{}:{}\n".format(arg, content)
    return s

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='SHOT')
    parser.add_argument('--gpu_id', type=str, nargs='?', default='0', help="device id to run")
    parser.add_argument('--s', type=int, default=0, help="source")
    parser.add_argument('--t', type=int, default=1, help="target")
    parser.add_argument('--max_epoch', type=int, default=20, help="max iterations")
    parser.add_argument('--batch_size', type=int, default=1024, help="batch_size")
    parser.add_argument('--worker', type=int, default=4, help="number of workers")
    parser.add_argument('--dset', type=str, default='office-home', choices=['VISDA-C', 'office', 'office-home', 'office-caltech'])
    parser.add_argument('--lr', type=float, default=1e-2, help="learning rate")
    parser.add_argument('--net', type=str, default='resnet50', help="vgg16, resnet50, resnet101")
    parser.add_argument('--seed', type=int, default=2020, help="random seed")
    parser.add_argument('--bottleneck', type=int, default=256)
    parser.add_argument('--epsilon', type=float, default=1e-5)
    parser.add_argument('--layer', type=str, default="wn", choices=["linear", "wn"])
    parser.add_argument('--classifier', type=str, default="bn", choices=["ori", "bn"])
    parser.add_argument('--smooth', type=float, default=0.1)   
    parser.add_argument('--output', type=str, default='san')
    parser.add_argument('--da', type=str, default='uda', choices=['uda', 'pda', 'oda'])
    parser.add_argument('--trte', type=str, default='val', choices=['full', 'val'])
    
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

    folder = './data/'
    args.s_dset_path = folder + args.dset + '/' + names[args.s] + '_list.txt'
    args.test_dset_path = folder + args.dset + '/' + names[args.t] + '_list.txt'     

    if args.dset == 'office-home':
        if args.da == 'pda':
            args.class_num = 65
            args.src_classes = [i for i in range(65)]
            args.tar_classes = [i for i in range(25)]
        if args.da == 'oda':
            args.class_num = 25
            args.src_classes = [i for i in range(25)]
            args.tar_classes = [i for i in range(65)]

    args.output_dir_src = osp.join(args.output, args.da, args.dset, names[args.s][0].upper())
    args.name_src = names[args.s][0].upper()
    if not osp.exists(args.output_dir_src):
        os.system('mkdir -p ' + args.output_dir_src)
    if not osp.exists(args.output_dir_src):
        os.mkdir(args.output_dir_src)

    args.out_file = open(osp.join(args.output_dir_src, 'log.txt'), 'w')
    args.out_file.write(print_args(args)+'\n')
    args.out_file.flush()
    train_source(args, dataset_path, dataset_info)

    args.out_file = open(osp.join(args.output_dir_src, 'log_test.txt'), 'w')
    for i in range(len(names)):
        if i == args.s:
            continue
        args.t = i
        args.name = names[args.s][0].upper() + names[args.t][0].upper()

        folder = '/Checkpoint/liangjian/tran/data/'
        args.s_dset_path = folder + args.dset + '/' + names[args.s] + '_list.txt'
        args.test_dset_path = folder + args.dset + '/' + names[args.t] + '_list.txt'

        if args.dset == 'office-home':
            if args.da == 'pda':
                args.class_num = 65
                args.src_classes = [i for i in range(65)]
                args.tar_classes = [i for i in range(25)]
            if args.da == 'oda':
                args.class_num = 25
                args.src_classes = [i for i in range(25)]
                args.tar_classes = [i for i in range(65)]

        test_target(args, dataset_path, dataset_info)