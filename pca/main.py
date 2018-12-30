#encoding=utf8
import numpy as np

#pca步骤
# 1.demean操作
# 2.算data的协方差矩阵
# 3.算协方差矩阵的特征值和特征向量
# 4.拿到特征值最大的k个特征向量组成映射矩阵P
# 5.data与P做线性变换得到pca后的data
def pca(data, k):
    #算每个特征的均值
    u = np.mean(data, axis=0)
    #deman
    after_demean = data - u
    #算协方差矩阵=(1/m)*(X^T*X)
    cov = (1/len(data))*after_demean.T.dot(after_demean)
    #算协方差矩阵的特征值和特征向量
    value, vector = np.linalg.eig(cov)
    #特征值排序，从大到小，因为特征值越大特征的方差越大
    idx = np.argsort(value)[::-1]
    idx = idx[:k]
    #特征值最大的k个特征向量组成映射矩阵P
    P = vector[idx]
    #降维
    return data.dot(P.T)


if __name__ == '__main__':
    data = np.array([[1, 1, 2, 4, 2], [1, 3, 3, 4, 4]])
    print(pca(data.T, 1))