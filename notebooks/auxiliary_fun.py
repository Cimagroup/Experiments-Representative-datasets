import numpy as np
import networkx as nx
from scipy.spatial import distance_matrix
import random
import math

def inf_distance_mat(x,X):
    m = X-x
    l = []
    for i in range(len(X)):
        l.append(np.max(abs(m[i])))
    return np.array(l)

def representative_distance(x,X):
    return np.min(inf_distance_mat(x,X))


def representative_point(j,X,Xrep): 
    m = distance_matrix(X,Xrep,p=np.inf)
    d = representative_distance(X[j],Xrep)
    return np.reshape(Xrep[m[j,:]==d],(3,))

    
def dominatingSet(X,y,epsilon=0.1):
    "Dominating dataset of X with a given labels y and representativeness factor epsilon."
    ady = distance_matrix(X,X,p=np.inf)
    g = nx.from_numpy_matrix(ady<epsilon)
    dom = nx.dominating_set(g)
    return np.array(list(dom))

def prod(w,x):
    "Product of a weight and a point of a dataset"
    return np.dot(w[1:],x)+w[0]

def norm(p):
    return np.max(abs(p))

def norm2(p):
    return math.sqrt(np.sum(p**2))

def distance(p1,p2):
    return norm2(p1-p2)
