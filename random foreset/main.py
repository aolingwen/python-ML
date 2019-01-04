from copy import deepcopy
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


class my_decisiontree(object):
    #算经验熵
    def calcInfoEntropy(self, feature, label):
        label_set = set(label)
        len_set = len(label_set)
        result = 0
        for l in label_set:
            count = 0
            for j in range(len(label)):
                if label[j] == l:
                    count += 1
            p = count/len(label)
            result -= p*np.log2(p)
        return result

    #算条件熵
    def calcHDA(self, feature, label, index, value):
        assert index < len(feature[0])
        count = 0
        sub_feature = []
        sub_label = []
        for i in range(len(feature)):
            if feature[i][index] == value:
                count+=1
                sub_feature.append(feature[i])
                sub_label.append(label[i])
        pHA = count/len(feature)
        e = self.calcInfoEntropy(sub_feature, sub_label)
        return pHA*e


    #算信息增益
    def calcInfoGain(self, feature, label, index):
        base_e = self.calcInfoEntropy(feature, label)
        f = np.array(feature)
        f_set = set(f[:, index])
        sum_HDA = 0
        for l in f_set:
            sum_HDA += self.calcHDA(feature, label, index, l)
        return base_e - sum_HDA


    #获得信息增益最高的特征
    def getBestFeature(self, feature, label):
        max_infogain = 0
        best_feature = 0
        for i in range(len(feature[0])):
            infogain = self.calcInfoGain(feature, label, i)
            if infogain > max_infogain:
                max_infogain = infogain
                best_feature = i
        return best_feature

    def createTree(self, feature, label):
        #样本里都是同一个label没必要继续分叉了
        if len(set(label)) == 1:
            return label[0]
        #样本中只有一个特征或者所有样本的特征都一样的话就看哪个label的票数高
        if len(feature[0]) == 1 or len(np.unique(feature, axis=0)) == 1:
            vote = {}
            for l in label:
                if l in vote.keys():
                    vote[l] += 1
                else:
                    vote[l] = 1
            max_count = 0
            ll = None
            for k, v in vote.items():
                if v > max_count:
                    max_count = v
                    ll = k
            return ll

        #根据信息增益拿到特征的索引
        best_feature = self.getBestFeature(feature, label)
        tree = {best_feature: {}}
        f = np.array(feature)
        #拿到bestfeature的所有特征值
        f_set = set(f[:, best_feature])
        #构建对应特征值的子样本集sub_feature, sub_label
        for v in f_set:
            sub_feature = []
            sub_label = []
            for i in range(len(feature)):
                if feature[i][best_feature] == v:
                    sub_feature.append(feature[i])
                    sub_label.append(label[i])
            #递归构建决策树
            tree[best_feature][v] = self.createTree(sub_feature, sub_label)
        return tree

    def __init__(self):
        self.tree = {}

    def fit(self, feature, label):
        self.tree = self.createTree(feature, label)

    def predict(self, feature):
        result = []
        def classify(tree, feature):
            if not isinstance(tree, dict):
                return tree
            t_index, t_value = list(tree.items())[0]
            f_value = feature[t_index]
            if isinstance(t_value, dict):
                #在随机森林预测时可能出现预测数据的特征值并没有出现在决策树里
                #所以在异常时，取与预测数据特征值最接近的值为特征值
                try:
                    classLabel = classify(tree[t_index][f_value], feature)
                except:
                    min_distance = 1e4
                    min_f = list(tree[t_index].keys())[0]
                    for k in tree[t_index].keys():
                        dis = abs(f_value-k)
                        if dis < min_distance:
                            min_distance = dis
                            min_f = k
                    classLabel = classify(tree[t_index][min_f], feature)

            return classLabel

        copy_tree = deepcopy(self.tree)
        # for f in feature:
        result.append(classify(copy_tree, feature))

        return np.array(result)



class my_rf(object):
    def __init__(self, num_tree):
        #num_tree为决策树的数量
        self._num_tree = num_tree
        #决策树列表，用来保存构建好的决策树
        self._tree_list = None
        #用来保存每棵树随机取到的特征的索引
        self._feature_idx_list = None

    def fit(self, feature, label):
        self._tree_list = [my_decisiontree() for _ in range(self._num_tree)]
        self._feature_idx_list = [[] for _ in range(self._num_tree)]
        for i in range(self._num_tree):
            #有放回抽样拿样本
            sample_idx = np.random.randint(0, len(feature), size=len(feature))
            sample_row = feature[sample_idx]
            sample_label = label[sample_idx]
            #随机选取一半的特征来构造决策树
            shuffle_idx = np.random.permutation(len(feature[0]))
            sample_row = sample_row[:, shuffle_idx]
            sample_row = sample_row[:, :int(len(feature[0])/2)]
            self._feature_idx_list[i] = shuffle_idx[:int(len(feature[0])/2)]

            self._tree_list[i].fit(sample_row, sample_label)



    def predict(self, feature):
        result = []
        for i, f in enumerate(feature):
            vote = {}
            for j in range(self._num_tree):
                tf = f[self._feature_idx_list[j]]
                r = self._tree_list[j].predict(tf)
                for l in r:
                    if l in vote.keys():
                        vote[l] += 1
                    else:
                        vote[l] = 1
            #bagging，取投票数最高的预测为结果
            max_count = 0
            ll = None
            for k, v in vote.items():
                if v > max_count:
                    max_count = v
                    ll = k
            result.append(ll)

        return np.array(result)



if __name__=='__main__':
    iris = datasets.load_iris()
    X = iris.data.astype(int)
    y = iris.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    lr = my_rf(10)
    lr.fit(X_train, y_train)
    predict = lr.predict(X_test)
    print(accuracy_score(y_test, predict))