import numpy as np
import pandas as pd

path = '/home/huangyuqing/tab-ddpm-main/data/abalone/'

pd.DataFrame(data=np.load(path+'X_cat_train.npy',allow_pickle=True)).to_csv(path+'X_cat_train.csv')
pd.DataFrame(data=np.load(path+'X_cat_val.npy',allow_pickle=True)).to_csv(path+'X_cat_val.csv')
pd.DataFrame(data=np.load(path+'X_cat_test.npy',allow_pickle=True)).to_csv(path+'X_cat_test.csv')
pd.DataFrame(data=np.load(path+'X_num_test.npy',allow_pickle=True)).to_csv(path+'X_num_test.csv')
pd.DataFrame(data=np.load(path+'X_num_train.npy',allow_pickle=True)).to_csv(path+'X_num_train.csv')
pd.DataFrame(data=np.load(path+'X_num_val.npy',allow_pickle=True)).to_csv(path+'X_num_val.csv')
pd.DataFrame(data=np.load(path+'y_test.npy',allow_pickle=True)).to_csv(path+'y_test.csv')
pd.DataFrame(data=np.load(path+'y_train.npy',allow_pickle=True)).to_csv(path+'y_train.csv')
pd.DataFrame(data=np.load(path+'y_val.npy',allow_pickle=True)).to_csv(path+'y_val.csv')
