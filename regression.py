import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression,Ridge
from sklearn.preprocessing import StandardScaler,PolynomialFeatures


# X = np.array([[60,1,15],[80,2,10],[100,3,8],[120,3,8],[150,4,2]])
# Y= np.array([[400],[600],[850],[950],[1300]])

# scaled = StandardScaler()
# X_Scaled = scaled.fit_transform(X)

# model = LinearRegression()
# model.fit(X_Scaled,Y)

# new_house = np.array([[100,2,2]])
# new_house_scaler = scaled.transform(new_house)
# y_pred = model.predict(new_house_scaler)
# print(y_pred)


# X = np.array([[1],[2],[3],[4],[5]])
# y = np.array([3, 6, 10, 12, 25])

# lin_model = LinearRegression()
# lin_model.fit(X, y)
# y_lin_pred = lin_model.predict(X)

# poly = PolynomialFeatures(degree=2)
# X_poly = poly.fit_transform(X)
# poly_model = LinearRegression()
# poly_model.fit(X_poly, y)
# X_fit = np.linspace(1,6).reshape(-1,1)
# y_poly_pred = poly_model.predict(poly.transform(X_fit))

# # رسم نمودار
# plt.scatter(X, y, color='blue', label='Data')
# plt.plot(X, y_lin_pred, color='green', linestyle='--', label='Linear Regression')
# plt.plot(X_fit, y_poly_pred, color='red', label='Polynomial Regression')
# plt.xlabel('Year')
# plt.ylabel('Car Price')
# plt.title('Linear vs Polynomial Regression')
# plt.legend()
# plt.grid(True)
# plt.show()



# X = np.array([20, 40, 60, 80, 100, 120, 140]).reshape(-1,1)
# Y = np.array([10, 7, 5, 5.5, 7, 9, 12])



# poly = PolynomialFeatures(degree=2)
# x_poly = poly.fit_transform(X)

# model = LinearRegression()
# model.fit(x_poly,Y)

# speed = np.array([[90]])
# predict_speed = model.predict(poly.transform(speed))

# print(f"مصرف پیش‌بینی‌شده برای سرعت 90: {predict_speed[0]:.2f} لیتر در 100km")


X = np.array([
    [60, 1, 0, 1],
    [80, 1, 1, 0],
    [100, 0, 1, 1],
    [120, 1, 1, 1],
    [140, 0, 0, 1],
    [160, 1, 1, 1],
    [180, 0, 1, 0]])

Y = np.array([90000, 130000, 150000, 180000, 210000, 240000, 270000])


model_ridge = Ridge(alpha=1)
model_ridge.fit(X,Y)

X_new = np.array([
    [110, 1, 1, 0],  # خانه 110 متر با آسانسور و پارکینگ ولی بدون انباری
    [150, 0, 1, 1]   # خانه 150 متر بدون آسانسور، با پارکینگ و انباری
])


Y_predict = model_ridge.predict(X_new)
print(Y_predict)