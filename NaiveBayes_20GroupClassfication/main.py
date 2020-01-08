import warnings
warnings.filterwarnings('ignore')
#Loading the data set - training data.
from sklearn.datasets import fetch_20newsgroups
twenty_train = fetch_20newsgroups(subset='train', shuffle=True)

print("The number of training examples", len(twenty_train.data))
# printing top five training examples
print(twenty_train.data[0:5])

# Danh sách tất cả tên các lớp
print(twenty_train.target_names)

targets = twenty_train.target
print('Nhãn các lớp {}'.format(targets))

print('Số nhãn: {}'.format(len(targets)))

#Hiển thị dòng đầu tiên của văn bản đầu tiên
print("\nDòng đầu tiên của văn bản đầu tiên:\n".join(twenty_train.data[0].split("\n")[:3]))

#Chuẩn bị dữ liệu huấn luyện
#Tạo ma trận term-document, trong đó giá trị ở mỗi ô là số lần xuất hiện của từ trong văn bản chứa nó
from sklearn.feature_extraction.text import CountVectorizer
count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(twenty_train.data)

print(X_train_counts.shape)
print(X_train_counts[0])


#Biểu diễn văn bản bằng TF-IDF
# TF-IDF
from sklearn.feature_extraction.text import TfidfTransformer
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

print(X_train_tfidf.shape)
print(X_train_tfidf[0])


    #Huấn luyện mô hình
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB

text_clf = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()), ('clf', MultinomialNB())])
text_clf = text_clf.fit(twenty_train.data, twenty_train.target)

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import ShuffleSplit


def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt


# estimator = MultinomialNB()
# title = "Learning Curves (Naive Bayes)"
# # Cross validation with 100 iterations to get smoother mean test and train
# # score curves, each time with 20% data randomly selected as a validation set.
#
# cv = ShuffleSplit(n_splits=100, test_size=0.2, random_state=0)
# X, y = X_train_tfidf, twenty_train.target
# plot_learning_curve(estimator, title, X, y, ylim=(0.0, 1.01), cv=cv, n_jobs=8)
# plt.show()


#Đánh giá mô hình trên dữ liệu test
import numpy as np
twenty_test = fetch_20newsgroups(subset='test', shuffle=True)
predicted = text_clf.predict(twenty_test.data)
np.mean(predicted == twenty_test.target)

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(twenty_test.target, predicted)
plt.figure(figsize=(10, 10))
plt.imshow(cm, cmap="Reds")
plt.show()
print(cm)


#
# #Cải tiến mô hình
# #Khởi tạo mô hình có dùng thêm tham số loại bỏ đi các từ dừng
# # Removing stop words
# from sklearn.pipeline import Pipeline
# text_clf = Pipeline([('vect', CountVectorizer(stop_words='english')), ('tfidf', TfidfTransformer()),
#                      ('clf', MultinomialNB())])
#
# import nltk
# # nltk.download('stopwords')
#
# print('steming the corpus... Please wait...')
# from nltk.stem.snowball import SnowballStemmer
# stemmer = SnowballStemmer("english", ignore_stopwords=True)
#
#
# class StemmedCountVectorizer(CountVectorizer):
#     def build_analyzer(self):
#         analyzer = super(StemmedCountVectorizer, self).build_analyzer()
#         return lambda doc: ([stemmer.stem(w) for w in analyzer(doc)])
#
#
# stemmed_count_vect = StemmedCountVectorizer(stop_words='english')
# text_mnb_stemmed = Pipeline([('vect', stemmed_count_vect), ('tfidf', TfidfTransformer()),
#                              ('mnb', MultinomialNB(fit_prior=False))])
# text_mnb_stemmed = text_mnb_stemmed.fit(twenty_train.data, twenty_train.target)
# predicted_mnb_stemmed = text_mnb_stemmed.predict(twenty_test.data)
# print('Result: {}'.format(np.mean(predicted_mnb_stemmed == twenty_test.target)))


#Trực quan hoá quá trình huấn luyện của NB và SVM

# estimator = MultinomialNB()
# title = "Learning Curves (Naive Bayes)"
# # Cross validation with 100 iterations to get smoother mean test and train
# # score curves, each time with 20% data randomly selected as a validation set.
#
# cv = ShuffleSplit(n_splits=100, test_size=0.2, random_state=0)
# X, y = X_train_tfidf, twenty_train.target
# plot_learning_curve(estimator, title, X, y, ylim=(0.0, 0.95), cv=cv, n_jobs=8)
#
