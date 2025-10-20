import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Lasso
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


url = 'https://raw.githubusercontent.com/mwaskom/seaborn-data/master/mpg.csv'
data = pd.read_csv(url).dropna()


x = data.drop('mpg',axis=1)
y = data['mpg']

numerice = ['cylinders','displacement','horsepower','weight','acceleration','model_year']
categories = ['origin','name']

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)

preprocessing = ColumnTransformer(
    transformers=(
       ( 'numerice',StandardScaler(),numerice),
       ('categories',OneHotEncoder(sparse_output=False,handle_unknown='ignore'),categories)
    )
)

nn_pipeline = Pipeline([
    ('preprocessor',preprocessing),
    ('model',Lasso(alpha=0.1))
])

nn_pipeline.fit(x_train,y_train)
y_predict =nn_pipeline.predict(x_test)
print(y_predict)

mae = mean_absolute_error(y_test, y_predict)
mse = mean_squared_error(y_test, y_predict)
r2 = r2_score(y_test, y_predict)

print("MAE:", mae)
print("MSE:", mse)
print("R²:", r2)

