import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder,StandardScaler

X = np.array([
    [150,7],[160,6],[150,8],[120,10],[130,9],[110,11],[950,20],[800,18]
])
Y = np.array(["apple", "apple", "apple", "plum", "plum", "plum", "watermelon", "watermelon"])

le = LabelEncoder()
Y_encoded = le.fit_transform(Y)
scaler = StandardScaler()
x_scaler = scaler.fit_transform(X)


model = KNeighborsClassifier(n_neighbors=2)

model.fit(x_scaler, Y_encoded)
new_fruit = np.array([[600,22]])
new_fruit_scaled = scaler.transform(new_fruit)
pred = model.predict(new_fruit_scaled)
print("Predicted fruit:", le.inverse_transform(pred)[0])

plt.scatter(X[:,0], X[:,1], c=Y_encoded, cmap='rainbow', edgecolor='k', s=100)
plt.scatter(new_fruit[:,0], new_fruit[:,1], c='black', marker='*', s=200, label='New Fruit')

x_min, x_max = X[:,0].min() - 50, X[:,0].max() + 50
y_min, y_max = X[:,1].min() - 1, X[:,1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 5),
                     np.arange(y_min, y_max, 0.1))
Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.contourf(xx, yy, Z, alpha=0.3, cmap='rainbow')
plt.xlabel("Weight (grams)")
plt.ylabel("Diameter (cm)")
plt.title("KNN Fruit Classification")
plt.legend()
plt.show()
