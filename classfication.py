
from matplotlib.colors import ListedColormap
import numpy as np
import matplotlib.pyplot as plt
from sklearn.calibration import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier,plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC


# X = np.array ([
#     [11,2],
#     [14,2],
#     [15,3],
#     [15.5,4],
#     [14,3],
#     [17,5],
#     [18,7],
#     [20,10]
# ])

# Y = np.array([0,0,0,0,0,1,1,1])
# model = LogisticRegression()
# model.fit(X,Y)

# # در آرایه ها دنبال اولین ستون از  داده ها می گردد
# x_min,x_max = X[:,0].min()-1,X[:,0].max()+1
# # در آرایه دنبال دومین ستون از داده ها
# y_min , y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
# # حالا ما میایم یک صفحه ی شطرنجی 100 در 100 درست می کنیم
# xx,yy = np.meshgrid(np.linspace(x_min,x_max,100),np.linspace(y_min,y_max,100))
# #حالا چون مدل لاجستیک رگرشن نمی تونه با دوتا مارتریس xx,yy کار کنه باید تبدیل به z کنیم.
# z = model.predict(np.c_[xx.ravel(),yy.ravel()])
# # بعد مقدار z  را به همان شکل قبلی x,y تبدیل می کنیم
# z = z.reshape(xx.shape)

# plt.contour(xx,yy,z,alpha=0.3,cmap=plt.cm.coolwarm)
# plt.scatter(X[:, 0], X[:, 1], c=Y, cmap=plt.cm.coolwarm, edgecolors='k')
# plt.ylabel("Hours of Study")
# plt.xlabel("Exam Score")
# plt.title("Logistic Regression Classification (Pass or Fail)")
# plt.show()


# X = np.array([
#   [1,40],
#   [2,45],
#   [2,50],
#   [3,55],
#   [4,60],
#   [5,65],
#   [6,70],
#   [7,75],
#   [8,80],
#   [9,85]
#  ])

# Y = np.array([0,0,0,0,1,1,1,1,1,1])
# model = LogisticRegression()
# model.fit(X,Y)
# new_students = np.array([[3,50],[7,78]])
# pridections = model.predict(new_students)
# print(pridections)


# ---DecisionTree


# X = np.array([
#     [150,7],[160,6],[170,8],[120,10],[130,9],[110,11],[950,14],[800,11]
# ])
# Y = np.array([0,0,0,1,1,1,2,2])
# tree  = DecisionTreeClassifier(max_depth=3,random_state=0)
# tree.fit(X,Y)

# new_fruites = np.array([[155,7],[900,10]])
# predictions = tree.predict(new_fruites)

# print(predictions)
# plt.figure(figsize=(8,6))
# plot_tree(tree, feature_names=['Weight', 'Sweetness'], class_names=['Apple','Plum','Hani'], filled=True)
# plt.show()

# ------------RandomForest

# X = np.array([
#     [140,4],[150,3],[200,5],
#     [50,2],[40,3],[10,1],
#     [1200,15],[1700,16],[2000,18]
# ])
# Y = np.array(["apple","apple","apple","plum","plum","plum","watermelon","watermelon","watermelon"])

# model = RandomForestClassifier(n_estimators=5,random_state=42)
# model.fit(X,Y)

# new_frouits = np.array([[400,6],])
# pred = model.predict(new_frouits)
# print(pred)


#-----K-NN
# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.preprocessing import LabelEncoder,StandardScaler

# X = np.array([
#     [150,7],[160,6],[150,8],[120,10],[130,9],[110,11],[950,20],[800,18]
# ])
# Y = np.array(["apple","apple","apple","alo","alo","alo","watermelon","watermelon"])

# le = LabelEncoder()
# Y_encoded = le.fit_transform(Y)
# scaler = StandardScaler()
# x_scaler = scaler.fit_transform(X)


# model = KNeighborsClassifier(n_neighbors=2)

# model.fit(x_scaler, Y_encoded)
# new_fruit = np.array([[600,22]])
# new_fruit_scaled = scaler(new_fruit)
# pred = model.predict(new_fruit_scaled)
# print("Predicted fruit:", le.inverse_transform(pred)[0])

# plt.scatter(X[:,0], X[:,1], c=Y_encoded, cmap='rainbow', edgecolor='k', s=100)
# plt.scatter(new_fruit[:,0], new_fruit[:,1], c='black', marker='*', s=200, label='New Fruit')

# x_min, x_max = X[:,0].min() - 50, X[:,0].max() + 50
# y_min, y_max = X[:,1].min() - 1, X[:,1].max() + 1
# xx, yy = np.meshgrid(np.arange(x_min, x_max, 5),
#                      np.arange(y_min, y_max, 0.1))
# Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
# Z = Z.reshape(xx.shape)

# plt.contourf(xx, yy, Z, alpha=0.3, cmap='rainbow')
# plt.xlabel("Weight (grams)")
# plt.ylabel("Diameter (cm)")
# plt.title("KNN Fruit Classification")
# plt.legend()
# plt.show()

# -----SVM
X = np.array([
    [150,7],[160,6],[150,8],[120,10],[130,9],[110,11],[950,20],[800,18]
])
Y = np.array(["apple","apple","apple","alo","alo","alo","watermelon","watermelon"])

encoder = LabelEncoder()
Y_encoder = encoder.fit_transform(Y)

scalerd = StandardScaler()
x_scaler = scalerd.fit_transform(X)

model = SVC()
model.fit(x_scaler,Y_encoder)
 
new_frite = np.array([[600,22]])
new_fruite_scaler = scalerd.transform(new_frite)
pred = model.predict(new_fruite_scaler)
print("Predicted fruit:", encoder.inverse_transform(pred)[0])




