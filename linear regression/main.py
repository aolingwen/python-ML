#encoding=utf8

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import datasets
from sklearn.metrics import r2_score


class my_linearRegression(object):
    def __init__(self):
        #W
        self.coef_ = None
        #b
        self.intercept_ = None
        #所有的W和b
        self._theta = None


    #正规方程解
    def fit_normal(self, X_train, y_train):
        assert X_train.shape[0] == y_train.shape[0]

        X_b = np.hstack([np.ones((len(X_train), 1)), X_train])
        self._theta = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y_train)

        self.intercept_ = self._theta[0]
        self.coef_ = self._theta[1:]

        return self


    #批量梯度下降求解
    def fit_gd(self, X_train, y_train, leraning_rate=1e-2, n_iters=1e4):
        assert X_train.shape[0] == y_train.shape[0]

        #mse loss
        def J(theta, X_b, y):
            try:
                return np.sum((y - X_b.dot(theta)) ** 2) / len(y)
            except:
                return float('inf')

        #算theta对loss的偏导
        def dJ(theta, X_b, y):
            res = np.empty(len(theta))
            #b对J的偏导
            res[0] = np.sum(X_b.dot(theta) - y)
            #w对J的偏导
            for i in range(1, len(theta)):
                res[i] = (X_b.dot(theta) - y).dot(X_b[:, i])
            return res * 2 / len(X_b)


        #批量梯度下降
        def gradient_descent(X_b, y, initial_theta, leraning_rate, n_iters=1e4, epsilon=1e-8):
            theta = initial_theta
            cur_iter = 0

            while cur_iter < n_iters:
                gradient = dJ(theta, X_b, y)
                last_theta = theta
                theta = theta - leraning_rate * gradient
                if (abs(J(theta, X_b, y) - J(last_theta, X_b, y)) < epsilon):
                    break
                cur_iter += 1
            return theta

        X_b = np.hstack([np.ones((len(X_train), 1)), X_train])
        initial_theta = np.zeros(X_b.shape[1])
        self._theta = gradient_descent(X_b, y_train, initial_theta, leraning_rate, n_iters)

        self.intercept_ = self._theta[0]
        self.coef_ = self._theta[1:]

        return self

    def predict(self, X_predict):
        #给定待预测数据集X_predict，返回表示X_predict的结果向量
        assert self.intercept_ is not None and self.coef_ is not None
        assert X_predict.shape[1] == len(self.coef_)

        X_b = np.hstack([np.ones((len(X_predict), 1)), X_predict])
        return X_b.dot(self._theta)


if __name__=='__main__':
    scaler = StandardScaler()
    boston = datasets.load_boston()
    data = boston.data
    target = boston.target
    target = target.reshape(-1, 1)
    show_data_target = np.hstack((data, target))
    boston_data = pd.DataFrame(show_data_target, columns=np.hstack((boston.feature_names,'PRICE')))
    target = target.reshape(1, -1)

    #把离群值去掉
    data = boston_data[boston_data['PRICE'] < 50].iloc[:, :-2]
    #数据归一化，如果不做归一化梯度下降效果很不理想
    data = scaler.fit_transform(data)
    target = boston_data[boston_data['PRICE'] < 50]['PRICE']

    train_data, test_data, train_price, test_price = train_test_split(data, target, test_size=0.2)


    line_reg = my_linearRegression()
    line_reg.fit_gd(train_data, train_price)
    predict = line_reg.predict(test_data)
    print(r2_score(test_price, predict))

