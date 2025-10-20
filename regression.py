# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# from sklearn.linear_model import Ridge
# from sklearn.metrics import mean_squared_error
# from sklearn.preprocessing import StandardScaler, OneHotEncoder
# from sklearn.compose import ColumnTransformer
# from sklearn.pipeline import Pipeline
# from sklearn.neural_network import MLPRegressor
# from sklearn.datasets import fetch_california_housing
# from sklearn.model_selection import train_test_split

# data = fetch_california_housing(as_frame=True)
# df = data.frame

# df.to_csv("california_housing.csv",index=False)

# data = pd.read_csv('california_housing.csv')

# X =  data.drop("MedHouseVal",axis=1)
# Y = data["MedHouseVal"]

# x_train,x_test,y_train,y_test = train_test_split(X,Y, test_size=0.2,random_state=42)

# nn_pipline = Pipeline(steps=[
#     ('scaler',StandardScaler()),
#     ('MLP',MLPRegressor(
#         hidden_layer_sizes=(64,32),
#         activation='relu',
#         solver=('adam'),
#         max_iter=1000,
#         random_state=42
#     ))
# ])

# nn_pipline.fit(x_train,y_train)

# y_pred = nn_pipline.predict(x_test)

# mse = mean_squared_error(y_test, y_pred)
# r2 = r2_score(y_test, y_pred)

# print(f"✅ MSE (میانگین خطا): {mse:.3f}")
# print(f"✅ R² (دقت مدل): {r2:.3f}")

# 🔟 رسم نمودار برای مقایسه
# plt.figure(figsize=(7,5))
# plt.scatter(y_test, y_pred, alpha=0.5)
# plt.xlabel("واقعی (Actual Price)")
# plt.ylabel("پیش‌بینی‌شده (Predicted Price)")
# plt.title("📈 Neural Network Regression - California Housing")
# plt.grid(True)
# plt.show()

# data = {
#     'YearsExperience': [1, 3, 5, 7, 10, 12, 15, 20],
#     'Education': ['Bachelor', 'Master', 'PhD', 'Bachelor', 'Master', 'PhD', 'Bachelor', 'PhD'],
#     'SkillsCount': [3, 5, 6, 4, 7, 8, 5, 9],
#     'Salary': [40000, 55000, 75000, 52000, 80000, 95000, 70000, 110000]
# }


# df = pd.DataFrame(data)

# X = df[["YearsExperience","Education","SkillsCount"]]
# Y = df["Salary"]

# encoder = OneHotEncoder(sparse_output=False)
# education_encoder = encoder.fit_transform(X[["Education"]])

# x_numeric = np.hstack([X.drop("Education",axis=True).values,education_encoder])

# scaled = StandardScaler()
# X_scaled = scaled.fit_transform(x_numeric)

# model = Ridge(alpha=1)
# model.fit(x_numeric,Y)


# new_employees = pd.DataFrame({
#     'YearsExperience': [4, 11, 16],
#     'Education': ['Bachelor', 'Master', 'PhD'],
#     'SkillsCount': [5, 7, 9]
# })

# edu_new_employee = encoder.transform(new_employees[["Education"]])
# x_edu_numeric = np.hstack((new_employees.drop("Education",axis=True).values,edu_new_employee))
# # X_new_numeric = np.hstack([new_employees[['YearsExperience', 'SkillsCount']].values, edu_new_encoded])


# predictions = model.predict(x_edu_numeric)
# print(predictions)

# #############

# data = {
#     'area': [70, 120, 90, 150, 200],
#     'rooms': [2, 3, 2, 4, 5],
#     'distance_to_center': [8, 6, 5, 3, 2],
#     'price': [180, 250, 220, 310, 400]
# }

# df = pd.DataFrame(data)

# X = df[["area","rooms","distance_to_center"]]
# Y = df[["price"]]

# Scaled = StandardScaler()
# x_scaled = Scaled.fit_transform(X)

# model = Ridge(alpha=1)
# model.fit(x_scaled,Y)

# new = pd.DataFrame({
#     'area': [300],
#     'rooms': [5],
#     'distance_to_center': [2],
# })

# new_scaled = Scaled.transform(new)
# predictions = model.predict(new_scaled)
# print(predictions)

# df = pd.DataFrame({
#     'brand': ['Apple', 'Samsung', 'Xiaomi', 'Apple', 'Samsung'],
#     'memory': [128, 256, 128, 64, 128],
#     'screen_size': [6.1, 6.7, 6.5, 5.8, 6.4],
#     'price': [999, 899, 499, 799, 699]
# })

# X = df[["brand","memory","screen_size"]]
# Y = df[["price"]]

# numeber = ['memory','screen_size']
# category = ['brand']

# preprossesor = ColumnTransformer(
#     transformers=[
#         ('number',StandardScaler(),numeber),
#         ('category',OneHotEncoder(),category)
#     ]
# )

# model = Pipeline(steps=[
#     ('preprocessing',preprossesor),
#     ('Ridge',Ridge(alpha=1))
# ])

# model.fit(X,Y)

# new = pd.DataFrame({
#     'brand': 'Apple',
#     'memory': [256],
#     'screen_size': [6.5],
# })

# predection = model.predict(new)
# print(predection)



# df = pd.DataFrame({
#     'study_hours': [1, 2, 3, 4, 5, 6, 7, 8],
#     'sleep_hours': [8, 7, 7, 6, 6, 5, 5, 4],
#     'school_type': ['public', 'public', 'private', 'private', 'public', 'private', 'public', 'private'],
#     'score': [60, 65, 70, 75, 72, 78, 80, 85]
# })

# X = df[["study_hours","sleep_hours","school_type"]]
# Y = df[["score"]]

# numerice_features = ['study_hours','sleep_hours']
# category_features = ['school_type']

# preprocessor = ColumnTransformer(
#     transformers=[
#         ('number',StandardScaler(),numerice_features),
#         ('category',OneHotEncoder(),category_features)
#     ]
# )

# model = Pipeline(steps=[
#     ('preprocess',preprocessor),
#     ('ridge',Ridge(alpha=1))
# ])

# model.fit(X,Y)

# new_student = pd.DataFrame({
#     'study_hours': [5],
#     'sleep_hours': [6],
#     'school_type': ['private']
# })

# predictions = model.predict(new_student)
# print(f"📘 پیش‌بینی نمره دانش‌آموز: {predictions[0]:.2f}")

# NN Regrestion
# data = pd.DataFrame({
#     "area": [60, 80, 100, 120, 150, 200, 250, 300, 350, 400],
#     "bedrooms": [1, 2, 2, 3, 3, 4, 4, 5, 5, 6],
#     "city": ["Paris", "Paris", "Lyon", "Lyon", "Nice", "Nice", "Marseille", "Marseille", "Lyon", "Paris"],
#     "price": [150, 200, 220, 260, 310, 400, 430, 460, 480, 520]
# })

# X = data[["area","bedrooms","city"]]
# Y = data["price"]

# numerice = ['area','bedrooms']
# category  = ['city']

# preprocessor = ColumnTransformer(
#     transformers=[
#         ('number',StandardScaler(),numerice),
#         ('catedgory',OneHotEncoder(),category)
#     ]
# )

# nnModel = MLPRegressor(
#     hidden_layer_sizes=(32,16),
#     activation=('relu'),
#     solver=('adam'),
#     max_iter=1000,
#     random_state=42
# )
# model = Pipeline(steps=[
#     ("preprocessor",preprocessor),
#     ('model',nnModel)
# ])
# model.fit(X,Y)

# new_data = pd.DataFrame({
#     'area': [90, 130, 210],
#     'bedrooms': [2, 3, 5],
#     'city':['Paris','Lyon','Marseille'],
# })


# prediction = model.predict(new_data)
# print(prediction)


