import pandas as pd
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import ElasticNet

url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/mpg.csv"

data = pd.read_csv(url).dropna()

x = data.drop("mpg",axis=1)
y = data['mpg']

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)

numeric = ['cylinders','displacement','horsepower','weight','acceleration']
categories = ['name','origin']

preprocessing = ColumnTransformer(
    transformers=[
        ('standard',StandardScaler(),numeric),
        ('oneHot',OneHotEncoder(sparse_output=False,handle_unknown='ignore'),categories)
    ]
)

elasticNet_pipeline = Pipeline([
    ( 'preprosseing',preprocessing),
    (  'model',ElasticNet(alpha=0.5))
])

elasticNet_pipeline.fit(x_train,y_train)
y_pred = elasticNet_pipeline.predict(x_test)

mae2 = mean_absolute_error(y_test, y_pred)
mse2 = mean_squared_error(y_test, y_pred)
r2_2 = r2_score(y_test, y_pred)

print("Model 1 → MAE:", mae2, " | MSE:", mse2, " | R²:", r2_2)
