
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression, Lasso
#from sklearn.select_model import cross_validation
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.covariance import EllipticEnvelope
from sklearn.decomposition import PCA
from sklearn.svm import SVR
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.preprocessing import normalize


def loadData(filename):
    data = np.genfromtxt(filename, delimiter=',', dtype=float, skip_header=1)
    return data


data = loadData("data/final_dataDec.csv")


def remove_missing_data(pX_train, feature_to_impute):
    X_train = np.copy(pX_train)
    for i in range(X_train.shape[0]-1, 0, -1):
        for j in range(0, X_train.shape[1], 1):
            if feature_to_impute[j] != 0 and X_train[i, j] == 0:
                X_train = np.delete(X_train, i, 0)
                break
    return X_train


impute = np.array([0] * len(data[0]))
impute[14] = 2 # num_bed
impute[15] = 2 # year_built
impute[18] = 2 # num_room
impute[19] = 2 # num_bath
impute[20] = 1 # living_space

data_removal = remove_missing_data(data, impute)

print('kich thuoc du lieu {}'.format(data_removal.shape))


def mean_imputation_pure(pX_train, feature_to_impute):
    X_train =  np.copy(pX_train)
    for i in range(0, len(feature_to_impute)):
        if feature_to_impute[i] == 0:
            continue
        non_zeros = 0
        for j in range(0, X_train.shape[0]):
            if X_train[j, i] != 0:
                non_zeros += 1
        mean = np.sum(X_train[:, i])/float(non_zeros)
        for j in range(0, X_train.shape[0]):
            if X_train[j, i] == 0:
                X_train[j, i] = mean
    return X_train


data_imputation = mean_imputation_pure(data, impute)

print('kich thuoc {}'.format(data_imputation.shape))


xrm = data_removal[:, :39]## Biến độc lập
yrm = data_removal[:, 39] # Biến phụ thuộc

from sklearn.model_selection import train_test_split
xrm_train, xrm_test, yrm_train, yrm_test = train_test_split(xrm, yrm, test_size=0.3, random_state=42)


#Hồi quy tuyến tính

reg_rm = LinearRegression()
reg_rm.fit(xrm_train, yrm_train)

print('do chinh xac {}'.format(reg_rm.score(xrm_train, yrm_train)))
print('mean absolute error {}'.format(mean_absolute_error(yrm_test, reg_rm.predict(xrm_test))))


#Hồi quy phi tuyến K lân cận gần nhất

knnreg_rm = KNeighborsRegressor(n_neighbors=50)
knnreg_rm.fit(xrm_train, yrm_train)

print('do chinh xac knn {}'.format(knnreg_rm.score(xrm_train, yrm_train)))
print('mean absolute error knn {}'.format(mean_absolute_error(yrm_test, knnreg_rm.predict(xrm_test))))

mae = []
for k in range(1, 50):
    reg = KNeighborsRegressor(n_neighbors=k)
    reg.fit(xrm_train, yrm_train)
    error = mean_absolute_error(yrm_test, reg.predict(xrm_test))
    mae.append(error)

import matplotlib.pyplot as plt
plt.plot(mae, c='red')
plt.show()

print("Optimal k: ", np.argmin(mae)+1)

