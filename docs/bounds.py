import perceptron as p
import math
import numpy as np


def prod(w,x):
    return np.dot(w[1:],x)+w[0]
    
def norm(p1):
    return np.max(abs(p1))
def norm1(p1):
    return np.sum(abs(p1))
#def distance(p1,p2):
#    return norm(p1-p2)

#def norm(p):
#    return np.sum(p**2)
def distance(p1,p2):
    return norm(p1-p2)




def sigmoid_der(m,x):
    return ((np.exp(-x))/(1+np.exp(-x))**2)**m

def rho(m,wx,wx_):
    z2 = max(wx,wx_)
    z1 = min(wx,wx_)
    if z1>math.log(m):
        return sigmoid_der(m,z1)
    elif z2<math.log(m):
        return sigmoid_der(m,z2)
    else:
        return sigmoid_der(m,math.log(m))
    
 
def closest_node(node, nodes):
    dist_2 = np.abs(nodes - node)
    l = []
    for i in range(len(dist_2)):
        l.append(np.max(dist_2[i]))
    l = np.array(l)
    return np.argmin(l)

def bound(X,X_,y,y_,w,eps):
    k = (1/len(X))*(norm1(w)*eps)
    t = []
    for i in range(len(X)):
        x = X[i]
        l = y[i]
        X_l = X_[y_==l]
        x_=X_l[closest_node(x,X_l)]
        t.append(2*rho(1,prod(w,x),prod(w,x_))+rho(2,prod(w,x),prod(w,x_)))
    t = np.array(t)
    return np.sum(k*t)
