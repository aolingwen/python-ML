import numpy as np
import math
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


class NaiveBayesClassifier(object):
    def __init__(self):
        '''
        self.label_prob表示每种类别在数据中出现的概率
        例如，{0:0.333, 1:0.667}表示数据中类别0出现的概率为0.333，类别1的概率为0.667
        '''
        self.label_prob = {}
        '''
        self.condition_prob表示每种类别确定的条件下各个特征出现的概率
        例如训练数据集中的特征为 [[2, 1, 1],
                              [1, 2, 2],
                              [2, 2, 2],
                              [2, 1, 2],
                              [1, 2, 3]]
        标签为[1, 0, 1, 0, 1]
        那么当标签为0时第0列的值为1的概率为0.5，值为2的概率为0.5;
        当标签为0时第1列的值为1的概率为0.5，值为2的概率为0.5;
        当标签为0时第2列的值为1的概率为0，值为2的概率为1，值为3的概率为0;
        当标签为1时第0列的值为1的概率为0.333，值为2的概率为0.666;
        当标签为1时第1列的值为1的概率为0.333，值为2的概率为0.666;
        当标签为1时第2列的值为1的概率为0.333，值为2的概率为0.333,值为3的概率为0.333;
        因此self.label_prob的值如下：     
        {
            0:{
                0:{
                    1:0.5
                    2:0.5
                }
                1:{
                    1:0.5
                    2:0.5
                }
                2:{
                    1:0
                    2:1
                    3:0
                }
            }
            1:
            {
                0:{
                    1:0.333
                    2:0.666
                }
                1:{
                    1:0.333
                    2:0.666
                }
                2:{
                    1:0.333
                    2:0.333
                    3:0.333
                }
            }
        }
        '''
        self.condition_prob = {}
    def fit(self, feature, label):
        '''
        对模型进行训练，需要将各种概率分别保存在self.label_prob和self.condition_prob中
        :param feature: 训练数据集所有特征组成的ndarray
        :param label:训练数据集中所有标签组成的ndarray
        :return: 无返回
        '''
        row_num = len(feature)
        col_num = len(feature[0])
        for c in label:
            if c in self.label_prob:
                self.label_prob[c] += 1
            else:
                self.label_prob[c] = 1

        for key in self.label_prob.keys():
            # 计算每种类别在数据集中出现的概率
            self.label_prob[key] /= row_num
            # 构建self.condition_prob中的key
            self.condition_prob[key] = {}
            for i in range(col_num):
                self.condition_prob[key][i] = {}
                for k in np.unique(feature[:, i], axis=0):
                    self.condition_prob[key][i][k] = 0

        for i in range(len(feature)):
            for j in range(len(feature[i])):
                if feature[i][j] in self.condition_prob[label[i]]:
                    self.condition_prob[label[i]][j][feature[i][j]] += 1
                else:
                    self.condition_prob[label[i]][j][feature[i][j]] = 1

        for label_key in self.condition_prob.keys():
            for k in self.condition_prob[label_key].keys():
                total = 0
                for v in self.condition_prob[label_key][k].values():
                    total += v
                for kk in self.condition_prob[label_key][k].keys():
                    #计算每种类别确定的条件下各个特征出现的概率
                    self.condition_prob[label_key][k][kk] /= total


    def predict(self, feature):
        '''
        对数据进行预测，返回预测结果
        :param feature:测试数据集所有特征组成的ndarray
        :return:
        '''
        result = []
        #对每条测试数据都进行预测
        for i, f in enumerate(feature):
            #可能的类别的概率
            prob = np.zeros(len(self.label_prob.keys()))
            ii = 0
            for label, label_prob in self.label_prob.items():
                #计算概率
                prob[ii] = label_prob
                for j in range(len(feature[0])):
                    prob[ii] *= self.condition_prob[label][j][f[j]]
                ii += 1
            #取概率最大的类别作为结果
            result.append(list(self.label_prob.keys())[np.argmax(prob)])
        return np.array(result)
        

if __name__=='__main__':
    iris = datasets.load_iris()
    X = iris.data[:100].astype('int')
    y = iris.target[:100]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    lr = NaiveBayesClassifier()
    lr.fit(X_train, y_train)
    predict = lr.predict(X_test)
    print(accuracy_score(y_test, predict))
