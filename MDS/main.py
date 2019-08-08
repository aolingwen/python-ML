import numpy as np
from sklearn.datasets import load_breast_cancer

def mds(data,d):
    '''
    input:data(ndarray):待降维数据
          d(int):降维后数据维数
    output:Z(ndarray):降维后的数据
    '''

    #计算dist2,dist2i,dist2j,dist2ij
    m,n = data.shape
    dist =np.zeros((m,m))
    disti = np.zeros(m)
    distj = np.zeros(m)
    B = np.zeros((m,m))
    for i in range(m):
        dist[i] = np.sum(np.square(data[i]-data),axis=1).reshape(1,m)
    for i in range(m):
        disti[i] = np.mean(dist[i,:])
        distj[i] = np.mean(dist[:,i])
    distij = np.mean(dist)
    #计算B
    for i in range(m):
        for j in range(m):            
            B[i,j] = -0.5*(dist[i,j] - disti[i] - distj[j] + distij)
    #矩阵分解得到特征值与特征向量
    lamda,V=np.linalg.eigh(B)
    #计算Z
    index=np.argsort(-lamda)[:d]
    diag_lamda=np.sqrt(np.diag(-np.sort(-lamda)[:d]))
    V_selected=V[:,index]
    Z=V_selected.dot(diag_lamda)

    return Z 
    
cancer = load_breast_cancer()
X,y = cancer['data'],cancer['target']

def mds_main(data,d):
    m,n = data.shape
    dist =np.zeros((m,m))
    disti = np.zeros(m)
    distj = np.zeros(m)
    B = np.zeros((m,m))
    for i in range(m):
        dist[i] = np.sum(np.square(data[i]-data),axis=1).reshape(1,m)
    for i in range(m):
        disti[i] = np.mean(dist[i,:])
        distj[i] = np.mean(dist[:,i])
    distij = np.mean(dist)
    for i in range(m):
        for j in range(m):            
            B[i,j] = -0.5*(dist[i,j] - disti[i] - distj[j] + distij)
    lamda,V=np.linalg.eigh(B)
    index=np.argsort(-lamda)[:d]
    diag_lamda=np.sqrt(np.diag(-np.sort(-lamda)[:d]))
    V_selected=V[:,index]
    Z=V_selected.dot(diag_lamda)
    return Z 

x1 = mds(X,2)
x2 = mds_main(X,2)

l1 = np.mean(abs(x1-x2)) 

print('降维前后的L1距离', l1)  