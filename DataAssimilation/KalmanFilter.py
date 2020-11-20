import numpy as np
from numpy.linalg import inv

def KalmanGain(P, C, R):
    tempInv = inv(R + np.dot(C,np.dot(P,C.transpose())))
    res = np.dot(P,np.dot(C.transpose(),tempInv))
    return res

def update_prediction(x, K, C, y):
    res = x + np.dot(K,(y - np.dot(C , x)))
    return res

def update_covariance_matrix(P, K, C):
    I = np.identity(len(K))
    res = np.dot((I - np.dot(K ,C)),P)
    return res

def prior_covariance_matrix(A, Q, P):
    res = np.dot(A,np.dot(P,A.transpose())) + Q
    return res

def covariance_matrix(X):
    means = np.array([np.mean(X, axis = 1)]).transpose()
    dev_matrix = X - means
    res = np.dot(dev_matrix, dev_matrix.transpose())
    return res

def update_train(state, pos, n_hours, train):   
    try:
        for i in range(0,n_hours):
            train[pos+i+1][n_hours-i-1] = state
    except:
        print('Error')
    return train