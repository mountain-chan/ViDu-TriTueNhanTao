import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer

cancer = load_breast_cancer()

print('cancer.keys():\n {}'.format(cancer.keys()))

print('Kích thước dữ liệu:\n {}'.format(cancer.data.shape))

print('Các thuộc tính: \n{}'.format(cancer.feature_names))

print('Các lớp: \n{}'.format(cancer.target_names))


# Chia dữ liệu
from sklearn.model_selection import train_test_split
x = cancer.data
y = cancer.target
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, stratify=y, random_state=42)

print(x_train.shape)
print(x_test.shape)

#Import DecisionTreeClassifier
from sklearn.tree import DecisionTreeClassifier
#Import accuracy_score
from sklearn.metrics import accuracy_score

# Cây quyết định max_depth là độ sâu của cây
#tree = DecisionTreeClassifier(random_state=42)
tree = DecisionTreeClassifier(max_depth=4, random_state=42)

#Bắt đầu training
tree.fit(x_train, y_train)

#Dự đoán trên tập test
y_pred = tree.predict(x_test)

print('\nĐộ chính xác tập huấn luyện: {:.4f}'.format(tree.score(x_train, y_train)))
print('Độ chính xác tập kiểm tra: {:.4f}'.format(tree.score(x_test, y_test)))

# Biểu thị cây phân loại
from sklearn.tree import export_graphviz

#muốn màu mè cho các nhánh thì setting filled=True, các góc của các nhánh bo tròn rounded=True.
export_graphviz(tree, out_file='tree_classifier.dot',
                feature_names=cancer.feature_names,
                class_names=cancer.target_names,
                filled=True, rounded=True)

# Chuyển file dot sang file ảnh
from subprocess import call
call("dot -Tpng tree_classifier.dot > tree_classifier.png", shell=True)

# #Hiển thị file ảnh
# from IPython.display import Image
# Image(filename='tree_classifier.png')

list_important = tree.feature_importances_
print(list_important)

# Import matplotlib
import matplotlib.pyplot as plt

features = cancer.feature_names
n = len(features)
plt.figure(figsize=(8, 10))
plt.barh(range(n), tree.feature_importances_)
plt.yticks(range(n), features)
plt.title('Muc do quan trong cac thuoc tinh')
plt.ylabel('Cac thuoc tinh')
plt.xlabel('Muc do')
plt.show()






