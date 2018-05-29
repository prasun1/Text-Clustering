
# coding: utf-8

# In[319]:


from scipy.sparse import csr_matrix
import pandas as pd
from scipy.sparse import * 
# df = pd.read_csv(
#     filepath_or_buffer='train_pr3.dat', header = None)
train_file = open('train_pr3.dat','r')
docs = []
for row in train_file:
    docs.append(row.rstrip().split(" "))
valslist1,valslist2 = [],[]
for h in docs:
    list2,list1 = [],[]
    for i in range(0,len(h),2):
        list1.append(h[i])
    for i in range(1,len(h),2):
        list2.append(h[i])
    valslist1.append(list1)
    valslist2.append(list2)
valslist1


# In[320]:


# valslist2 = frequency
from array import array
import numpy as np
import pandas as pd
import scipy.sparse as sp
from numpy.linalg import norm
from collections import Counter, defaultdict
from scipy.sparse import csr_matrix

def build_matrix(docs):
    # list_main,list_id = []* len(docs),[]*len(docs)
    nrows = len(docs)
    idx = {}
    tid = 0
    nnz = 0
    # nnz = count
    for d in valslist1:
        nnz += len(d)
    print(nnz)
    # nnz = len(valslist1)
    #     print(list_main)   
    rows = len(docs)

    ind = np.zeros(nnz, dtype=np.int)
    val = np.zeros(nnz, dtype=np.int)
    ptr = np.zeros(nrows+1, dtype=np.int)
    i = 0  # document ID / row counter
    n = 0  # non-zero counter
    k = 0
    h = 0
    m = 0
    ptr[0] = 0
    #     # transfer values
    for d in valslist1:
        for i in range(len(d)):
    #         print("hi",d[i])
        # #     print(len(d),d[0])
            ind[k] = d[i]
            k += 1
        ptr[m+1] = k
        m +=1
    #     print("k",k)
    for d in valslist2:
        for i in range(len(d)):
    #         print("hi",d[i])
        # #     print(len(d),d[0])
        #     for i in range(0,len(d)):
            val[h] = d[i]
            h += 1
    mat = csr_matrix((val, ind, ptr), dtype=np.double)
    return mat,val,ind,ptr


# In[321]:


mat,val,ind,ptr = build_matrix(docs)


# In[322]:


print(ind[:10])


# In[323]:


mat = csr_matrix((val, ind, ptr), dtype=np.double)
print(mat)


# In[324]:


def csr_l2normalize(mat, copy=True, **kargs):
    if copy is True:
        mat = mat.copy()
    nrows = mat.shape[0]
    nnz = mat.nnz
    ind, val, ptr = mat.indices, mat.data, mat.indptr
    # normalize
    for i in range(nrows):
        rsum = 0.0    
        for j in range(ptr[i], ptr[i+1]):
            rsum += val[j]**2
        if rsum == 0.0:
            continue  # do not normalize empty rows
        rsum = 1.0/np.sqrt(rsum)
        for j in range(ptr[i], ptr[i+1]):
            val[j] *= rsum
            
    if copy is True:
        return mat


# In[325]:


mat2 = csr_l2normalize(mat, copy=True)
print(len(mat2.data))


# amd

# In[283]:


# from sklearn.manifold import TSNE
# X_embedded = TSNE(n_components=2).fit_transform(mat)
print(mat2)


# In[284]:


print(ind[0:10])


# In[285]:


# tsvd.explained_variance_ratio_[0:3].sum
X_sparse_tsvd[0][0]


# In[334]:


from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import TruncatedSVD
from scipy.sparse import csr_matrix
from sklearn import datasets
import numpy as np
tsvd = TruncatedSVD(n_components=150)
X_sparse_tsvd = tsvd.fit(mat2).transform(mat2)


# In[335]:



print(len(X_sparse_tsvd.data))


# In[337]:


# checked for len(X_sparse_tsvd), 15, 5, 3.   tried elbow method for the best result but
#    didn't get it. Maybe It doesn't work on Sparse data
from sklearn.neighbors import NearestNeighbors
nbrs = NearestNeighbors(n_neighbors=7).fit(X_sparse_tsvd)
distances, indices = nbrs.kneighbors(X_sparse_tsvd)


# In[338]:


import pandas as pd
t = pd.DataFrame(distances)
f = t.iloc[:,-1]
len(distances[56])


# In[339]:


a = sorted(f, reverse = True)


# In[340]:


len(a)


# In[341]:

# plotting graph to find eps
import matplotlib.pyplot as plt
plt.plot(a)
plt.ylabel('eps')
plt.xlabel('number of points')
plt.show()


# In[294]:


X_sparse_tsvd[3][1]-X_sparse_tsvd[2][1]


# In[342]:


def euclidean(x,y):
    return numpy.sqrt(numpy.sum((x-y)**2))


# In[344]:


 import numpy
 D = X_sparse_tsvd
 eps = 0.33
 list6,list7 = [],[]
 MinPts=7
 clusters = [0]*len(D)
 #Finding core and noise point by iterating to every number and see if it comes within
 #eps if yes then it is core and if not it is noise
 for i in range(0, len(D)):
     if (clusters[i] == 0):
         neighbors = []
         for n in range(0, len(D)):
             #defining proximity between points
             list6.append(euclidean(D[i],D[n]))
             if euclidean(D[i],D[n]) < eps:
                 neighbors.append(n)
         points = neighbors
         if(len(points) < MinPts): # it means it is a noise point
             clusters[i] = -1 # noise point
 print(clusters)
#from sklearn.cluster import DBSCAN
#db = DBSCAN(eps=0.33, min_samples=7).fit(X_sparse_tsvd)


# In[346]:


clusters = db.labels_
plt.plot(clusters)
plt.xlabel('number of points')
plt.show()


# In[347]:


clusters



# In[348]:


result = pd.DataFrame(clusters)
result.to_csv('pr3_dbscan_finaltry.dat',index=False,header=None)

