import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline

url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv'
columns = ['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigree','Age','Outcome']
data = pd.read_csv(url,names=columns)

x=data.drop('Outcome',axis=1)
y=data['Outcome']

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)

nn_model = Pipeline([
    ('standard',StandardScaler()),
    ('model',MLPRegressor(
        hidden_layer_sizes=(64,32),
        activation='relu',
        solver='adam',
        max_iter=1000,
        random_state=42
    ))

])
nn_model.fit(x_train,y_train)
y_pred = nn_model.predict(x_test)
print(y_pred)