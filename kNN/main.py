#encoding=utf8
import numpy as np
import math
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

class my_knn_classifer(object):
    def __init__(self, k=5):
        self._k = k
        self._train_datas = None
        self._train_labels = None

    #knn没训练过程
    def fit(self, train_datas, train_labels):
        self._train_datas = np.array(train_datas)
        self._train_labels = np.array(train_labels)

    #预测
    def predict(self, test_datas):
        assert test_datas.shape[1] == self._train_datas.shape[1]

        def _predict(test_data):
            # 算欧式距离
            distances = [math.sqrt(np.sum((test_data - vec) ** 2)) for vec in self._train_datas]
            # 排序
            nearest = np.argsort(distances)
            # 拿最近的K个label
            topK = [self._train_labels[i] for i in nearest[:self._k]]
            votes = {}
            result = None

            # 统计得票最多的label
            max_count = 0
            for label in topK:
                if label in votes.keys():
                    votes[label] += 1
                    if votes[label] > max_count:
                        max_count = votes[label]
                        result = label
                else:
                    votes[label] = 0
            return result

        #批量预测，比如测试集有100条数据，那每次都会调一次_predict生成每条数据的预测结果
        predict_result = [_predict(test_data) for test_data in test_datas]
        return np.array(predict_result)


if __name__=='__main__':
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target
    X_train, y_train, X_test, y_test = train_test_split(X, y, test_size=0.3)
    knn = my_knn_classifer()
    knn.fit(X_train, X_test)
    predict = knn.predict(y_train)
    print(accuracy_score(y_test, predict))




