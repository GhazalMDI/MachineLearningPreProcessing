import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/mpg.csv"
data = pd.read_csv(url)

data = data.dropna()

x = data.drop(['mpg'],axis=1)
y = data['mpg']

numerice = ["cylinders","horsepower", "weight", "acceleration", "model_year"]
category = ["origin","name"]

preprocessor = ColumnTransformer(
    transformers=[
        ('Processor',StandardScaler(),numerice),
        ('categories',OneHotEncoder(sparse_output=False,handle_unknown='ignore'),category)
    ]
)

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)

nn_pipline = Pipeline([
        ('preprosesor',preprocessor),
        ('nn_model',MLPRegressor(
            hidden_layer_sizes=(64,32),
            activation='relu',
            solver='adam',
            max_iter=1000,
            random_state=42
        ))
])

nn_pipline.fit(x_train,y_train)
y_pred = nn_pipline.predict(x_test)
print(y_pred)

mae1 = mean_absolute_error(y_test, y_pred)
mse1 = mean_squared_error(y_test, y_pred)
r2_1 = r2_score(y_test, y_pred)

print("Model 1 → MAE:", mae1, " | MSE:", mse1, " | R²:", r2_1)
