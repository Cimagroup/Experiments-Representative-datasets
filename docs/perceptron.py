import numpy as np
import random
import time
import auxiliary_fun as a
import bounds as b

def sigmoid(t):
    return 1/(1+np.exp(-t))
def sigmoid_derivative(p):
    return p*(1-p)

def loss_or(X,Labels,perceptron):
        l = 0
        for i in range(len(X)):
            l+=loss(X[i],Labels[i],perceptron.weight)
        l = (l/len(X))
        return l

## PERCEPTRON
    
def loss(x,c,w):
    wx = np.dot(w[1:],x)+w[0]
    return 0.5*(c-sigmoid(wx))**2


def loss_der(x,c,w,j):
    wx = np.dot(w[1:],x)+w[0]
    if j==0:
        return (c-sigmoid(wx))*(sigmoid(wx)-sigmoid(wx)**2)
    else:
        return (c-sigmoid(wx))*(sigmoid(wx)-sigmoid(wx)**2)*x[j-1]
            

class Perceptron:
    "Definition of the perceptron. It can be trained with stochastic gradient descent or gradient descent."
    def __init__(self,Xor,yor,lr=0.1):
        self.lr = lr
        self.weight = np.array([0.5]*4) # Change depending on input size.
        self.history= []
        self.history_or = []
        self.data_or = Xor
        self.label_or = yor
    def output(self,x):
        w = self.weight
        wx = np.dot(w[1:],x)+w[0]
        output = sigmoid(wx)
        return output
    def update_weight(self,x,c):
        w = self.weight
        for j in range(0,len(self.weight)):
            self.weight[j] = w[j]+self.lr*loss_der(x,c,w,j)
        print(self.weight)
    def update_weight_Full_dataset(self,X,Labels):
        w = self.weight
        for j in range(0,len(self.weight)):
            s = 0
            for i in range(len(X)):
                s+= loss_der(X[i],Labels[i],w,j)
            self.weight[j] = w[j]+(2/len(X))*self.lr*s
    def accuracy(self,X,Labels):
        "Measure of the accuracy over a dataset (X,Labels)."
        l = []
        for i in range(len(X)):
            l.append(self.output(X[i]))
        l = np.array(l)
        return np.sum((l<0.5)==(np.reshape(Labels,len(X))<0.5)) / len(X)

    def train(self,X,Labels,iterations = 10,stochastic = False):
        print('TRAINING')
        if stochastic:
            "Stochastic gradient descent using one point per iteration."
            l = range(len(X))
            j = 0            
            while j< iterations:
                print('Iteration number :',j," of ",iterations)
                print('Accuracy =', self.accuracy(X,Labels))
                self.history.append(self.accuracy(X,Labels))
                self.history_or.append(self.accuracy(self.data_or,self.label_or))
                j+=1
                i = random.choice(l)
                print('X_',i)
                self.update_weight(X[i],Labels[i])
        else:
            "Gradient descent."
            j = 0
            start = time.time()
            while j< iterations:
                print('Iteration number :',j," of ",iterations)
                print('Accuracy =', self.accuracy(X,Labels))
                self.history.append(self.accuracy(X,Labels))
                self.history_or.append(self.accuracy(self.data_or,self.label_or))
                j+=1
                self.update_weight_Full_dataset(X,Labels)
            end = time.time()
            print("Time of execution: ",end - start)
                
    def ordered_train(self,X,Labels):
        "Training of the perceptron using a dataset in order."
        for i in range(len(X)):
            print('Accuracy =', self.accuracy(X,Labels))
            print(self.weight)
            self.update_weight(X[i],Labels[i])
    def history(self):
        return self.history 
    
def output_over_dataset(X,p):
    "X is a dataset and p a perceptron."
    l = []
    for i in range(len(X)):
        l.append(sigmoid(a.prod(p.weight,X[i])))
    l = np.array(l)
    M = np.zeros((len(X),np.shape(X)[1]+1))
    M[:,:-1] = X
    M[:,-1] = l
    return M
