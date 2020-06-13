
import numpy as np
import random
import math
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
###############################################################
def gradient_descent (X_train, Y_train, l_rate, itr_num):
    n , m = X_train.shape
    W = np.zeros((m,1))
    #print(W)
    #print(Y_train)

    for  i in range (itr_num):
        Y_est =np.dot(X_train, W)
        diff=Y_est -Y_train
        cost =np.sum(diff**2)/(n*2.0)
        #print (cost)
        step_size =np.dot(X_train.transpose(),diff)
        #print(sum(diff))
        #print(step_size)
        W = W -step_size*(l_rate/float(n))
        m_deriv = 0
        b_deriv = 0
        m=0
        b=0
        for i in range(n):
        # Calculate partial derivatives
        # -2x(y - (mx + b))
            m_deriv += -2*X[i] * (Y[i] - (m*X[i] + b))
        # -2(y - (mx + b))
            b_deriv += -2*(Y[i] - (m*X[i] + b))
    print(m_deriv)
    print(b_deriv)
    print(step_size)        
    return W
def update_weights_Huber(m, b, X, Y, delta, learning_rate):
    m_deriv = 0
    b_deriv = 0
    N = len(X)
    for i in range(N):
        # derivative of quadratic for small values and of linear for large values
        if abs(Y[i] - m*X[i] - b) <= delta:
          m_deriv += -X[i] * (Y[i] - (m*X[i] + b))
          b_deriv += - (Y[i] - (m*X[i] + b))
        else:
          m_deriv += delta * X[i] * ((m*X[i] + b) - Y[i]) / abs((m*X[i] + b) - Y[i])
          b_deriv += delta * ((m*X[i] + b) - Y[i]) / abs((m*X[i] + b) - Y[i])
    
    # We subtract because the derivatives point in direction of steepest ascent
    m -= (m_deriv / float(N)) * learning_rate
    b -= (b_deriv / float(N)) * learning_rate

    return m, b
def huber_loss_cost(vec1,vec2, delta):
    N = len(vec1)
    dif = abs(vec1-vec2)
    comp = dif < delta
    term1=.5*dif**2
    #print(term1)
    term2=delta*(dif -delta/2)
    np.sum(term1*comp + term2*(1-comp))/float(N)     
def gradient_descent_huper_loss (X_train, Y_train, l_rate, itr_num, delta):
    n , m = X_train.shape
    W = np.zeros((m,1))
    m = 0
    b = 0
    print(X_train)
    print(Y_train)
    for  i in range (itr_num):
        Y_est =np.dot(X_train, W)
        dif=Y_est -Y_train
        dif_abs = abs(dif) 
        comp = dif_abs < delta
        x_train_transpose = X_train.transpose()
        term = comp*dif +(1-comp)*delta*dif/dif_abs
        step_size = np.dot(x_train_transpose, term)
        ###############################################################################
        #############################################
        m_deriv = 0
        b_deriv = 0
        
        for i in range(n):       
            if abs(Y_train[i][0] - m*X_train[i][0] - b) <= delta:
                m_deriv += -X_train[i][0] * (Y_train[i] - (m*X_train[i][0] + b))
                b_deriv += - (Y_train[i][0] - (m*X_train[i][0] + b))                
            else:
                m_deriv += delta * X_train[i][0] * ((m*X_train[i][0] + b) - Y_train[i]) / abs((m*X_train[i][0] + b) - Y_train[i])
                b_deriv += delta * ((m*X_train[i][0] + b) - Y_train[i]) / abs((m*X_train[i][0] + b) - Y_train[i])                
        # We subtract because the derivatives point in direction of steepest ascent
        #print(step_size)
        #print(m_deriv)
        #print(b_deriv)
        m -= m_deriv / float(n) * l_rate
        b -= b_deriv / float(n) * l_rate
        ###############################################################################
        ###############################################################################
        W = W -l_rate*step_size/float(n)
    print(m)
    print(b)
    print(W)
    return W 


#    retuen cost
num_vec= range(10,30,2)
cost_vec=[]
#cost_vec_mont=[]
index =0
#mont_num =10000
sample_num = 12
X = np.array([[i for i in range(sample_num)]]) 
X = X.transpose()
n, m =X.shape
X0 = np.ones((sample_num,1))
Xnew=np.hstack((X,X0))
#print(Xnew)
W = [[4],[-1]]
Y = np.dot(Xnew,W)
#print(Y) 
noise = np.random.normal(0,1,(sample_num,1)) 
Ynew = Y +noise
X_train, X_test, Y_train, Y_test = train_test_split( 
        Xnew, Ynew, test_size = 0.15)
W_0 = gradient_descent (X_train, Y_train, .001,1)
W_1 = gradient_descent_huper_loss(X_train, Y_train, .001,7000,1)
Y_est = np.dot(X_test, W_0)
diff=Y_est -Y_test
cost =np.sum(diff**2)/(n*2.0)
#print(abs(W_0-W_1))
#print((1<2)*3)
#cost_summation = cost + cost_summation
    #cost_vec.append(cost_summation/mont_num)
#plt.plot(cost_vec) 
#plt.show()
#print(W_0)  
#print(cost_vec)  

 
