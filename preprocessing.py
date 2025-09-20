
from sklearn import preprocessing
import numpy as np

X = np.array([[1.,-1.,2.],
              [2.,0.,0.],
              [0.,1.,-1.]])

scaler = preprocessing.StandardScaler().fit(X)
# X_scaled = scaler.transform(X)
# print(X_scaled)

print(preprocessing.StandardScaler().fit_transform(X))