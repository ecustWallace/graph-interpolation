# -%-coding:utf-8-%-

from utils import *
from preprocessing import *
import configuration as cfg
from predict import *
import os
from sklearn.externals import joblib

# Load data
data = Data()

# Build Laplacian Matrix
sparse_matrix = df2sparsematrix(data.train_data, data.n_user, data.n_item)
laplacian = build_laplacian(sparse_matrix)

# Prediction
filename = os.path.join('prediction','{}_{}_{}_{}_prediction.pkl'.format(cfg.FLAGS.db_name, cfg.FLAGS.filename, cfg.FLAGS.similarity, cfg.FLAGS.type))
if os.path.exists(filename):
    prediction = joblib.load(filename)
else:
    prediction= predict(sparse_matrix, laplacian) # This costs a long time
    joblib.dump(prediction, filename)

# Calculate RMSE from test data
rmse = get_rmse(prediction, data.test_data)
print(rmse)