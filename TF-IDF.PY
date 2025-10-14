# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.preprocessing import normalize
# from sklearn.metrics.pairwise import cosine_similarity
# import pandas as pd

# data=[
#     "The cat is sleeping on the sofa",
#     "A dog is playing in the garden",
#     "I love reading books about science",
#     "The garden has many colorful flowers",
#     "She is studying computer science at university"
# ]

# vectorizer = TfidfVectorizer(stop_words="english")
# X = vectorizer.fit_transform(data)
# x_norm = normalize(X,norm="l2")
# x_cos= cosine_similarity(x_norm)
# dats = pd.DataFrame(x_cos)
# print(dats)

####################

# from sklearn.feature_extraction.text import CountVectorizer
# import pandas as pd

# docs = ["I love data", "I love machine learning", "Data is powerful"]

# vecto = CountVectorizer(stop_words="english")
# X = vecto.fit_transform(docs)

# df = pd.DataFrame(X.toarray(),columns=vecto.get_feature_names_out())
# print(df)

####################

# from sklearn.cluster import KMeans
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt

# data = np.array([
#     [20, 2000],
#     [22, 2200],
#     [25, 2500],
#     [40, 8000],
#     [42, 7500],
#     [45, 9000]
# ])

# kmeans = KMeans(n_clusters=2,random_state=42)
# kmeans.fit(data)
# labels = kmeans.labels_
# centers = kmeans.cluster_centers_

# print(labels)
# print(centers)

# from sklearn.preprocessing import OneHotEncoder
# import pandas as pd

# data_train = [
#     ['male', 'US', 'Mobile'],
#     ['female', 'Europe', 'Desktop'],
#     ['female', 'Asia', 'Mobile']
# ]

# data_test = [
#     ['male', 'US', 'Tablet'],       
#     ['female', 'Africa', 'Mobile'],  
#     ['female', 'Europe', 'Desktop']  
# ]

# enc = OneHotEncoder(drop="first",handle_unknown="ignore",sparse_output=False)
# enc.fit(data_train)

# x_train_enc = enc.transform(data_train)
# x_test_enc = enc.transform(data_test)


# decode = enc.inverse_transform(x_test_enc)

# df_train = pd.DataFrame(data_train, columns=['gender', 'region', 'device'])
# df_train_enc = pd.DataFrame(x_train_enc, columns=enc.get_feature_names_out())

# df_test = pd.DataFrame(data_test, columns=['gender', 'region', 'device'])
# df_test_enc = pd.DataFrame(x_test_enc, columns=enc.get_feature_names_out())
# df_decoded = pd.DataFrame(decode, columns=['gender', 'region', 'device'])

# print("Encoded Train:")
# print(df_train_enc)
# print("\nEncoded Test:")
# print(df_test_enc)
# print("\nDecoded Test:")
# print(df_decoded)


from sklearn.preprocessing import OneHotEncoder
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

df_train  = pd.DataFrame([
    ["male","A",'Yes',1],
    ["female","B","No",0],
    ["female","O","No",0],
    ["male","AB","Yes",1]
],columns=["gender","blood_type","smoker","needs checkup"])

x_train = df_train[["gender","blood_type","smoker"]]
y_train = df_train["needs checkup"]

df_test = pd.DataFrame([
    ['female', 'A', 'no'],
    ['male', 'O', 'yes'],
    ['female', 'C', 'no']  
],columns=['gender', 'blood_type', 'smoker'])

pipline = Pipeline([
    ("encoder",OneHotEncoder(drop="first",handle_unknown='ignore',sparse_output=False)),
    ("model",DecisionTreeClassifier())
])
pipline.fit(x_train,y_train)

pred = pipline.predict(df_test)
print(pred)