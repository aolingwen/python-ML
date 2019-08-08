import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from collections import Counter

def AGNES(feature, k):
    '''
    AGNES聚类并返回聚类结果
    假设数据集为`[1, 2], [10, 11], [1, 3]]，那么聚类结果可能为`[[1, 2], [1, 3]], [[10, 11]]]
    :param feature:训练数据集所有特征组成的ndarray
    :param k:表示想要将数据聚成`k`类，类型为`int`
    :return:聚类结果
    '''
    # 找到距离最小的下标
    def find_Min(M):
        min = np.inf
        x = 0;
        y = 0
        for i in range(len(M)):
            for j in range(len(M[i])):
                if i != j and M[i][j] < min:
                    min = M[i][j];
                    x = i;
                    y = j
        return (x, y, min)

    #计算簇间最大距离
    def calc_max_dist(cluster1, cluster2):
        max_dist = 0
        for i in range(len(cluster1)):
            for j in range(len(cluster2)):
                dist = np.sqrt(np.sum(np.square(cluster1[i] - cluster2[j])))
                if dist > max_dist:
                    max_dist = dist
        return max_dist

    #初始化C和M
    C = []
    M = []
    for i in feature:
        Ci = []
        Ci.append(i)
        C.append(Ci)
    for i in C:
        Mi = []
        for j in C:
            Mi.append(calc_max_dist(i, j))
        M.append(Mi)
    q = len(feature)
    #合并更新
    while q > k:
        x, y, min = find_Min(M)
        C[x].extend(C[y])
        C.pop(y)
        M = []
        for i in C:
            Mi = []
            for j in C:
                Mi.append(calc_max_dist(i, j))
            M.append(Mi)
        q -= 1
    return C
    
    
iris = datasets.load_iris()
X = iris.data
y = iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.6, random_state=2)
pred = AGNES(X_train, 3)

cluster = [[], [], []]
for i in range(len(X_train)):
    find = False
    for k in range(len(pred)):
        for p in range(len(pred[k])):
            if (X_train[i] == pred[k][p]).all():
                cluster[k].append(y_train[i])
                find = True
                break

        if find:
            break

score = 0
for k in range(len(cluster)):
    v = sorted(Counter(cluster[k]).items(), key=lambda x: x[1], reverse=True)
    true_label = v[0][0]
    score += np.sum(np.array(cluster[k]) == true_label) / len(cluster[k])
score /= len(cluster)
print(score)