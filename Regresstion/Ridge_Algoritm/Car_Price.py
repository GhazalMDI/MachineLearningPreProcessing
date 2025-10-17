from sklearn.preprocessing import OneHotEncoder,LabelEncoder
from sklearn.linear_model import Ridge
import numpy as np
import pandas as pd


data = {
    'Year': [2015, 2017, 2014, 2018, 2016, 2013, 2019],
    'KM': [50000, 30000, 80000, 20000, 40000, 90000, 10000],
    'Fuel': [0, 0, 1, 0, 1, 0, 0],  
    'Transmission': [0, 1, 0, 1, 1, 0, 1],  
    'Doors': [4, 4, 4, 4, 2, 2, 4],
    'Brand': ['Pride', 'BMW', 'Benz', 'BMW', 'Benz', 'Pride', 'BMW'],
    'Price': [12000, 25000, 30000, 28000, 27000, 10000, 32000]
}

df = pd.DataFrame(data)
X  = df[["Year","KM","Fuel","Transmission","Doors","Brand"]]
Y  = df[["Price"]]

encoder = OneHotEncoder(sparse_output=False)
brand_encoded = encoder.fit_transform(X[['Brand']])

x_numeric = np.hstack([X.drop('Brand',axis=1).values,brand_encoded])


model = Ridge(alpha=1)
model.fit(x_numeric,Y)


new_cars = pd.DataFrame({
    'Year': [2017, 2015, 2016],
    'KM': [25000, 60000, 50000],
    'Fuel': [0, 0, 1],
    'Transmission': [1, 0, 1],
    'Doors': [4, 4, 4],
    'Brand': ['BMW', 'Pride', 'Benz']
})

brand_new_encoded = encoder.transform(new_cars[['Brand']])
x_new_numeric=np.hstack([new_cars.drop('Brand',axis=1).values,brand_new_encoded])

prediction = model.predict(x_new_numeric)
print(prediction)