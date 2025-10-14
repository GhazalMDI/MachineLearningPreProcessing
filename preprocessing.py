
# from sklearn.datasets import make_classification
# from sklearn.linear_model import LogisticRegression
# from sklearn.model_selection import train_test_split
# from sklearn.pipeline import make_pipeline
# from sklearn.preprocessing import StandardScaler

# x,y = make_classification(random_state=42)
# x_train,x_test,y_train,y_test = train_test_split(x,y,random_state=42)
# pipe = make_pipeline(StandardScaler(),LogisticRegression())
# pipe.fit(x_train,y_train)

# pipe.score(x_test,y_test)
# print(pipe.score(x_test,y_test))

# import numpy as np
# from sklearn import preprocessing
# x_train = np.array([
#     [1.,-1.,2.],
#     [2.,0.,0.],
#     [0.,1.,-1,]
# ])

# min_max_scaler  = preprocessing.MinMaxScaler(x_train)
# x_train_minmax = min_max_scaler.fit_transform(x_train)
# print(x_train_minmax)

# x_test = np.array([[-3.,-1.,4.]])
# x_test_minmax = min_max_scaler.transform(x_test)

# import numpy as np
# from sklearn import preprocessing

# x_train = np.array([
#     [1.,-1.,2.],
#     [2.,0.,0.],
#     [0.,1.,-1,]
# ])

# max_abs_scaler = preprocessing.MaxAbsScaler(x_train)
# x_train_maxabs = max_abs_scaler.fit_transform(x_train)

# x_test = np.array([[-3.,-1.,4.]])
# x_test_maxbs = max_abs_scaler.transform(x_test)

# from sklearn import preprocessing
# from sklearn.datasets import load_iris
# from sklearn.model_selection import train_test_split
# import numpy as np



# دیتاست
# X, Y = load_iris(return_X_y=True)
# x_train, x_test, y_train, y_test = train_test_split(X, Y, random_state=0)

# quantile_transformer = preprocessing.QuantileTransformer(output_distribution='normal',random_state=0)

# x_train_trans = quantile_transformer.fit_transform(x_train)

# x_test_trans = quantile_transformer.transform(x_test)

# print(np.percentile(x_train[:, 0], [0, 25, 50, 75, 100]))
# print(np.percentile(x_train_trans[:, 0], [0, 25, 50, 75, 100]))

# X= [[1.,-1.,2.],
#     [2.,0.,0.],
#     [0.,1.,-1.]]
# x_normalize = preprocessing.normalize(X,norm='l2')
# normalizer = preprocessing.Normalizer()
# normalizer.transform([[-1.,  1., 0.]])
# print(normalizer.transform([[-1.,  1., 0.]]))

