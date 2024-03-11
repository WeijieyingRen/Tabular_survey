import argparse
import tent
import torch
import json
from torch import nn
from torch.utils.data import Dataset
from sklearn.datasets import load_iris
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
import pdb
import os
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score
from sklearn import metrics
from tab_transformer_pytorch import FTTransformer

def read_info(info_path):
    with open(info_path) as f:
        f_list = []
        for line in f:
            tokens = line.strip().split()
            f_list.append(tokens)
    #return f_list[:-1], int(f_list[-1][-1])
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

class FT_Model(nn.Module):
    def __init__(self, class_num, FTTrans, num_idx, cat_idx):
        super(FT_Model, self).__init__()
        self.tabTrans = FTTrans
        self.bn = nn.BatchNorm1d(48)
        self.dense = nn.Linear(in_features=48, out_features=class_num, bias=False)
        self.relu=torch.nn.ReLU()
        self.sm = nn.Softmax(dim=1)
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
        x = self.bn(x)
        x = self.relu(x)
        x = self.dense(x)
        x = self.sm(x)
        return x

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

class myDataset(Dataset):
    def __init__(self, data, label):
        self.data = data 
        self.label = label
    
    def __getitem__(self,index):
        return self.data[index], self.label[index]

    def __len__(self):
        return len(self.data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='TENT')
    parser.add_argument('--datapath', type=str, default='../data/')
    parser.add_argument('--dataset', type=str, default='airbnb', choices=['ASSISTments','Hospital_Readmission','Sepsis',
                                                                               'airbnb','channel','jigsaw','wine'])
    parser.add_argument('--mask_option', type=str, default='Gaussian', choices=['Gaussian', 'Uniform', 'Mask'])
    parser.add_argument('--model', type=str, default='mlp', choices=['mlp', 'FTTrans'])
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    
    dataset_path = os.path.join(args.datapath, args.dataset + '/')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    with open(os.path.join(dataset_path, 'info.json')) as f:
        dataset_info = json.load(f)
    
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

    data_train = myDataset(torch.from_numpy(X_train).to(torch.float32), torch.from_numpy(y_train).to(torch.long))
    data_test = myDataset(torch.from_numpy(X_test).to(torch.float32), torch.from_numpy(y_test).to(torch.long))
    data_test_ood = myDataset(torch.from_numpy(X_test_ood).to(torch.float32), torch.from_numpy(y_test_ood).to(torch.long))

    torch.manual_seed(args.seed)
    data_loader_train = torch.utils.data.DataLoader(dataset=data_train,
                                                    batch_size = args.batch_size,
                                                    shuffle = True)
    
    if args.model == 'mlp':
        model = Model(dataset_info['n_num_features']+dataset_info['n_cat_features'], dataset_info['class_num']).to(device)
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
        model = FT_Model(dataset_info['class_num'], FTTrans, dataset_info['num_index'], dataset_info['cat_index']).to(device)

    cost = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())

    for epoch in range(args.epochs):
        sum_loss=0
        train_correct=0
        for data in data_loader_train:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            outputs=model(inputs).squeeze()
            optimizer.zero_grad()
            loss=cost(outputs,labels)
            loss.backward()
            optimizer.step()
    
            mask = outputs.argmax(1)
            sum_loss+=loss.data
            train_correct+=torch.sum(mask==labels.data)

        print('[%d,%d] loss:%.03f' % (epoch + 1, args.epochs, sum_loss / len(data_loader_train)))
        print('        correct:%.03f%%' % (100 * train_correct / len(data_train)))

    model = tent.configure_model(model)
    params, param_names = tent.collect_params(model)
    optimizer = torch.optim.Adam(params, lr=1e-3)
    tented_model = tent.Tent(model, optimizer)

    tented_model.eval()
    batch = 64
    for i in range(int(len(data_test.data)/batch)):
        if i == 0:
            y_pred = tented_model(data_test.data[i*batch:(i+1)*batch].to(device))
        else:
            y_pred = torch.cat([y_pred, model(data_test.data[i*batch:(i+1)*batch].to(device))],dim=0)
    y_pred = torch.cat([y_pred, model(data_test.data[int(len(data_test.data)/batch)*batch:].to(device))],dim=0)
    mask = y_pred.argmax(1).cpu()
    accuracy = metrics.accuracy_score(data_test.label, mask)          
    f1_score = metrics.f1_score(data_test.label, mask, average='macro')

    print ('iid accuracy is: {}'.format(accuracy))
    print ('iid f1 score is:{}'.format(f1_score))
        

    for i in range(int(len(data_test_ood.data)/batch)):
        if i == 0:
            y_pred_ood = tented_model(data_test_ood.data[i*batch:(i+1)*batch].to(device))
        else:
            y_pred_ood = torch.cat([y_pred_ood, tented_model(data_test_ood.data[i*batch:(i+1)*batch].to(device))],dim=0)
    y_pred_ood = torch.cat([y_pred_ood, tented_model(data_test_ood.data[int(len(data_test_ood.data)/batch)*batch:].to(device))],dim=0)
    mask_ood = y_pred_ood.argmax(1).cpu()
    accuracy = metrics.accuracy_score(data_test_ood.label, mask_ood)          
    f1_score = metrics.f1_score(data_test_ood.label, mask_ood, average='macro')

    print ('ood accuracy is: {}'.format(accuracy))
    print ('ood f1 score is:{}'.format(f1_score))
