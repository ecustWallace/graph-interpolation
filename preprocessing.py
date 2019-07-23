# -%-coding:utf-8-%-
import numpy as np
from sklearn import preprocessing as pre
from sklearn.externals import joblib
import os
from tqdm import tqdm
import configuration as cfg

def df2sparsematrix(data, n_user, n_item):
    matrix = np.zeros([n_user, n_item], dtype='float')
    for index, row in enumerate(data.values.tolist()):
      matrix[int(row[0]-1),int(row[1]-1)] = float(row[2])
    matrix[matrix==0] = np.nan
    return matrix

def build_laplacian(sparse_matrix):
    filename = os.path.join('graph', '{}_{}_{}_{}.pkl'.format(cfg.FLAGS.db_name, cfg.FLAGS.filename, cfg.FLAGS.similarity, cfg.FLAGS.type))
    if os.path.exists(filename):
        print(filename + ' has already been trained.')
        laplacian = joblib.load(filename)
    else:
        print('{} laplacian is training'.format(cfg.FLAGS.similarity))
        if cfg.FLAGS.type == 'item_based':
            sparse_matrix = np.transpose(sparse_matrix)
        if cfg.FLAGS.similarity == 'cosine':
            laplacian = _cosine_(sparse_matrix, cfg.FLAGS.min_common)
        elif cfg.FLAGS.similarity == 'corr':
            laplacian = _corr_(sparse_matrix, cfg.FLAGS.min_common)
        else:
            raise(NotImplemented)
        joblib.dump(laplacian,filename)
        print(filename + ' has been constructed.')
    return laplacian

def _cosine_(sparse_matrix, min_common):
    laplacian = np.zeros([sparse_matrix.shape[0], sparse_matrix.shape[0]])
    for i in tqdm(range(sparse_matrix.shape[0])):
        for j in range(i):
            sparse_i, sparse_j= sparse_matrix[i,:].copy(), sparse_matrix[j,:].copy()
            common_idx = ~np.isnan(sparse_i * sparse_j)
            if np.sum(common_idx) < min_common:
                laplacian[i,j], laplacian[j,i] = 0, 0
                continue
            sparse_i[~common_idx] = np.nan
            sparse_j[~common_idx] = np.nan
            sparse_i[np.isnan(sparse_i)] = 0
            sparse_j[np.isnan(sparse_j)] = 0
            laplacian[i, j] = np.sum(sparse_i * sparse_j / np.linalg.norm(sparse_i) / np.linalg.norm(sparse_j))
            laplacian[j, i] = laplacian[i, j]
    degree_matrix = np.diag(np.sum(laplacian, axis=0))
    laplacian = degree_matrix - laplacian
    return laplacian

def _corr_(sparse_matrix):
    pass