import numpy as np
import random
from sklearn.model_selection import train_test_split
###############################################################
def gradient_descent (X_train, Y_train, l_rate, itr_num):
    n , m = X_train.shape
    W = np.zeros((m,1))
    for  i in range (itr_num):
        Y_est =np.dot(X_train, W)
        diff=Y_est -Y_train
        cost =np.sum(diff**2)/(n*2.0)
        print (cost)
        step_size =np.dot(X_train.transpose(),diff)/n
        shape1=(X_train.transpose()).shape
        shape2=diff.shape
        W = W -l_rate*step_size
    return W
# testing the Gradient Descent in linear regression in 2D space. 
X = np.array([[i for i in range(11)]]) 
X = X.transpose()
n, m =X.shape
X0 = np.ones((11,1))
Xnew=np.hstack((X,X0))
W = [[4],[-1]]
Y = np.dot(Xnew,W)
print(Y) 
noise = np.random.normal(0,.5,(11,1)) 
Ynew = Y +noise
X_train, X_test, Y_train, Y_test = train_test_split( 
             Xnew, Ynew, test_size = 0.15)
W_0 =gradient_descent (X_train, Y_train, .001,20000)
print(W_0)    
