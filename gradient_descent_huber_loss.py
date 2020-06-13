
import numpy as np
import random
import math
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
###############################################################
def gradient_descent (X_train, Y_train, l_rate, itr_num):
    n , m = X_train.shape
    W = np.zeros((m,1))
    for  i in range (itr_num):
        Y_est =np.dot(X_train, W)
        diff=Y_est -Y_train
        cost =np.sum(diff**2)/(n*2.0)
        step_size =np.dot(X_train.transpose(),diff)
        W = W -step_size*(l_rate/float(n))
   return W
def huber_loss_cost(vec1,vec2, delta):
    N = len(vec1)
    dif = abs(vec1-vec2)
    comp = dif < delta
    term1=.5*dif**2
    term2=delta*(dif -delta/2)
    np.sum(term1*comp + term2*(1-comp))/float(N)     
def gradient_descent_huper_loss (X_train, Y_train, l_rate, itr_num, delta):
    n , m = X_train.shape
    W = np.zeros((m,1))
    for  i in range (itr_num):
        Y_est =np.dot(X_train, W)
        dif=Y_est -Y_train
        dif_abs = abs(dif) 
        comp = dif_abs < delta
        x_train_transpose = X_train.transpose()
        term = comp*dif +(1-comp)*delta*dif/dif_abs
        step_size = np.dot(x_train_transpose, term)
        W = W -l_rate*step_size/float(n)
  return W 


index =0
sample_num = 12
X = np.array([[i for i in range(sample_num)]]) 
X = X.transpose()
n, m =X.shape
X0 = np.ones((sample_num,1))
Xnew=np.hstack((X,X0))
W = [[4],[-1]]
Y = np.dot(Xnew,W)
noise = np.random.normal(0,1,(sample_num,1)) 
Ynew = Y +noise
X_train, X_test, Y_train, Y_test = train_test_split( 
        Xnew, Ynew, test_size = 0.15)
W_0 = gradient_descent (X_train, Y_train, .001,1000)
W_1 = gradient_descent_huper_loss(X_train, Y_train, .001,3000,1)


 
