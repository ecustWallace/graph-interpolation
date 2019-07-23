# -%-coding:utf-8-%-
import numpy as np
import time
from tqdm import tqdm

def predict(sparse_matrix, laplacian):
    print('Prediction starts.')
    # Eliminate all rows and columns with diagonal element 0
    idx = np.diag(laplacian) != 0 # Effective Users
    laplacian = laplacian[idx, :]
    laplacian = laplacian[:, idx]
    # Initialize
    predicted_matrix = np.zeros(sparse_matrix.shape)
    theta = np.nanmean(sparse_matrix)
    # Step 1
    n_laplacian, _ = _normalize_(laplacian)
    # Step 2
    laplacian_square = np.dot(n_laplacian, n_laplacian)
    for i in tqdm(range(sparse_matrix.shape[0])): # Scan each user
        du_signal = sparse_matrix[i,idx]
        known_idx = ~np.isnan(du_signal)
        unknown_idx = np.isnan(du_signal)
        du_signal[np.isnan(du_signal)] = 0
        laplacian_sc = laplacian_square[~known_idx, :]
        laplacian_sc = laplacian_sc[:, ~known_idx]
        e_value, _ = np.linalg.eig(laplacian_sc)
        sorted_indices = np.argsort(e_value)
        W = np.sqrt(e_value[sorted_indices[0]])
        # Step 3
        laplacian_updated = _bilateral_update_(laplacian, du_signal, theta)
        # laplacian_updated = laplacian.copy()
        # Step 4
        n_laplacian_updated, degree_matrix = _normalize_(laplacian_updated)
        e_value, e_vector = np.linalg.eig(n_laplacian_updated)
        sorted_indices = np.argsort(e_value)
        e_value = e_value[sorted_indices]
        e_vector = e_vector[:, sorted_indices]
        # Step 5
        K = np.sum(e_value <= W)
        if K>1:
            pass
        # Step 6
        g = np.dot(degree_matrix**(1/2), du_signal)
        # Step 7
        u_k = e_vector[:,:K]
        u_ksc = u_k[~known_idx, :]
        u_ks = u_k[known_idx, :]
        g_sc = np.linalg.multi_dot([u_ksc, np.linalg.pinv(np.dot(np.transpose(u_ks), u_ks)), np.transpose(u_ks), g[known_idx]])
        g[unknown_idx] = g_sc
        # Step 8
        predicted_matrix[i,idx] = np.dot(np.diag(np.diag(degree_matrix)**(-1/2)), g)
        # if np.sum(np.isnan(predicted_matrix[i,:]))>0:
        #    raise Exception('{} prediction exists nan'.format(i))
    predicted_matrix[predicted_matrix<0] = 0
    predicted_matrix[predicted_matrix>5] = 5
    print('Prediction finishes')
    prediction = {}
    prediction['predicted_matrix'] = predicted_matrix
    prediction['idx'] = idx
    return prediction

def get_rmse(prediction, test_data):
    movie_idx = np.where(prediction['idx'])[0] + 1
    predicted_matrix = prediction['predicted_matrix']
    data = np.array(test_data, dtype='int16')
    rmse = 0
    effective_sample = 0
    for i in range(data.shape[0]):
        if data[i,1] not in movie_idx:
            continue
        else:
            effective_sample += 1
            rmse = rmse + (predicted_matrix[data[i,0]-1,np.where(movie_idx == data[i,1])[0][0]] - data[i,2])**2
    rmse = (rmse / effective_sample) ** (1/2)
    return rmse

def _normalize_(laplacian):
    degree_matrix = np.diag(np.diag(laplacian))
    degree_matrix_half = np.diag(np.diag(degree_matrix) ** (-1/2))
    degree_matrix_half[degree_matrix_half==np.inf] = 0
    norm_laplacian = np.linalg.multi_dot([degree_matrix_half, laplacian, degree_matrix_half])
    return norm_laplacian, degree_matrix

def _bilateral_update_(laplacian, du_signal, theta):
    idx = np.where(du_signal != 0)
    idx = idx[0]
    adj = laplacian.copy() # Adjacency Matrix
    adj = -adj + np.diag(np.diag(adj))
    for j in range(idx.shape[0]):
        for k in range(j):
            diff = np.abs(du_signal[idx[j]] - du_signal[idx[k]])
            adj[j, k] = adj[j,k] * np.exp(-diff**2/(theta**2))
            adj[k, j] = adj[j,k]
    new_diag = np.sum(adj, axis = 0)
    laplacian = -adj + np.diag(new_diag)
    return laplacian