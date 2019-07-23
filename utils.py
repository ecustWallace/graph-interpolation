# -%-coding:utf-8-%-

import os
from urllib.request import urlretrieve
import zipfile
import numpy as np
import pandas as pd
import random
import configuration as cfg

class Data():
    def __init__(self):
        download_data(cfg.FLAGS.db_name)
        self.train_data, self.test_data = load(cfg)
        self.n_user, self.n_item = getinfo(cfg.FLAGS.db_name)


def download_data(db_name):
    if not os.path.exists(os.path.join('data',db_name)):
        print('downloading {} dataset'.format(db_name))
        url = 'http://files.grouplens.org/datasets/movielens/' + db_name +'.zip'
        urlretrieve(url, db_name+'.zip')
        with zipfile.ZipFile(db_name+'.zip') as zf:
            zf.extractall('data')
        os.remove(db_name+'.zip')
        print('downloading finished')
    else:
        print('{} dataset exists'.format(db_name))

def load(cfg):
    data = pd.DataFrame(columns=['user','item','rating'])
    train_name = os.path.join('data',cfg.FLAGS.db_name,cfg.FLAGS.filename+'.base')
    test_name = os.path.join('data',cfg.FLAGS.db_name,cfg.FLAGS.filename+'.test')
    train_data = pd.read_csv(train_name, header = None, sep='\t', names=['user','item','rating','timestamp']).astype('float')
    test_data = pd.read_csv(test_name, header = None, sep='\t', names=['user','item','rating','timestamp']).astype('float')
    train_data = train_data.iloc[:,0:-1]
    test_data = test_data.iloc[:,0:-1]
    return train_data, test_data

def getinfo(db_name):
    info_file = os.path.join('data',db_name,'u.info')
    with open(info_file) as f:
        n_user = int(f.readline().split(' ')[0])
        n_item = int(f.readline().split(' ')[0])
    return n_user, n_item

def split_data(data, seed = 3, rate = 0.8):
    random.seed(seed)
    arr = np.arange(data.shape[0])
    random.shuffle(arr)
    n = int(rate * data.shape[0])
    train_idx = arr[0:n]
    test_idx = arr[n:]
    train_data = data.iloc[train_idx, :]
    test_data = data.iloc[test_idx, :]
    return train_data, test_data