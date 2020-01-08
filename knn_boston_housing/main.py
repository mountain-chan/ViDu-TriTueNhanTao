import pandas as pd
import numpy as np

boston = pd.read_csv('data/Boston.csv')
print(boston.head())

boston = boston.drop('Unnamed: 0', axis=1)
print(boston.head())

y = boston['medv'].values
x = boston.drop('medv', axis=1).values


# xrm=boston['rm']
xrm = x[:, 5]

xrm = xrm.reshape(-1, 1)
y = y.reshape(-1, 1)

import matplotlib.pyplot as plt
plt.scatter(xrm, y)
plt.ylabel("y: Value of house / 1000 USD")
plt.xlabel("x: Number of rooms")
plt.show()


from sklearn.linear_model import LinearRegression
reg = LinearRegression()
reg.fit(xrm, y)
# Hệ số R^2
print(reg.score(xrm, y))

xx = np.linspace(min(xrm), max(xrm)).reshape(-1, 1)
plt.scatter(xrm, y, color="blue")
plt.plot(xx, reg.predict(xx), color="red", linewidth=3)
plt.ylabel("y: Value of house / 1000 USD")
plt.xlabel("x: Number of rooms")
plt.show()

from yellowbrick.regressor import ResidualsPlot
visualizer = ResidualsPlot(reg, hist=False)

visualizer.fit(xrm, y)  # Fit the training data to the model
visualizer.score(xrm, y)  # Evaluate the model on the test data
visualizer.poof()


# Dự báo giá nhà dựa vào tất cả các biến

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)
reg = LinearRegression()
reg.fit(x_train, y_train)
y_pred = reg.predict(x_test)
# Hệ số R^2
print(reg.score(x_train, y_train))

from yellowbrick.regressor import ResidualsPlot
viz = ResidualsPlot(reg, hist=False)

viz.fit(x_train, y_train)  # Fit the training data to the model
viz.score(x_test, y_test)  # Evaluate the model on the test data
viz.poof()


# Kiểm tra độ chính xác của mô hình


plt.scatter(xrm, y)
plt.ylabel("y: Value of house / 1000 USD")
plt.xlabel("x: Number of rooms")
plt.show()


#Hồi quy dựa vào K lân cận gần nhất (Kneighbors)

from sklearn.neighbors import KNeighborsRegressor

# reg = KNeighborsRegressor(n_neighbors=9, weights='distance')
reg = KNeighborsRegressor(n_neighbors=1)
reg.fit(xrm, y)

xx = np.linspace(min(xrm), max(xrm)).reshape(-1, 1)
plt.scatter(xrm, y, color="blue")
plt.plot(xx, reg.predict(xx), color="red", linewidth=3)
plt.ylabel("y: Value of house / 1000 USD")
plt.xlabel("x: Number of rooms")
plt.show()


print("Test set R^2 k=1: {:.2f}".format(reg.score(xrm, y)))

# Test với k = 5
reg = KNeighborsRegressor(n_neighbors=5)
reg.fit(xrm, y)

xx = np.linspace(min(xrm), max(xrm)).reshape(-1, 1)
plt.scatter(xrm, y, color="blue")
plt.plot(xx, reg.predict(xx), color="red", linewidth=3)
plt.ylabel("y: Value of house / 1000 USD")
plt.xlabel("x: Number of rooms")
plt.show()


#Lựa chọn số  k tốt nhất

from sklearn.model_selection import GridSearchCV

params = {'n_neighbors': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]}

reg = KNeighborsRegressor()
model = GridSearchCV(reg, params, cv=5)
model.fit(xrm, y)
print('the best k: {}'.format(model.best_params_))

# k = 9
reg = KNeighborsRegressor(n_neighbors=9)
reg.fit(xrm, y)

xx = np.linspace(min(xrm), max(xrm)).reshape(-1, 1)
plt.scatter(xrm, y, color="blue")
plt.plot(xx, reg.predict(xx), color="red", linewidth=3)
plt.ylabel("y: Value of house / 1000 USD")
plt.xlabel("x: Number of rooms")
plt.show()

print("Test set R^2 k=9: {:.2f}".format(reg.score(xrm, y)))

#Dự báo giá nhà với tất cả các biến
reg = KNeighborsRegressor(n_neighbors=1)
reg.fit(x_train, y_train)
reg_pred = reg.predict(x_test)
print('Độ chính xác du bao tat ca cac bien: {}'.format(reg.score(x_test, y_test)))



