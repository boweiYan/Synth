import math
import random
import numpy as np
import numpy.random as rn
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import metrics
import gnrlBayesCl


def array2str(nparray):
    str1=''
    for i in range(nparray.shape[0]):
        str1 += ' '+str(nparray[i])
    return str1

def Hmean_ex(TP,TN):
    # Define HMean as a function to TP and TN, restricted P(true)=0.5
    return 4*TP*TN/(TP+TN)

if __name__=='__main__':
    eta = np.array([0.49,0.5,0.51])
    mu = np.array([0.3,0.4,0.3])
    best = -1
    bestTP = -1
    bestTN = -1
    f = np.zeros(3)
    #(bestC,bestS)=gnrlBayesCl.best_classifier(eta,mu,2,3,metrics.HMean)
    # Test all randomized classifiers
    N=1
    for i in range(N+1):
        for j in range(N+1):
            for k in range(N+1):
                f=np.array([i,j,k])*1./N
                (TP,FP,FN) = gnrlBayesCl.binaryMetrics(eta,mu,f)
                print i,j,k
                print TP,FP,FN,1-TP-FP-FN
                (loss,g1,g2,g3)=metrics.HMean(TP,FP,FN)
                print g1,g2,g3,loss
                print ((g1-g2-g3)*eta+g2)*mu
                print '\n'

    # Now let's check the optimal condition
    best1 = np.array([0,1,0])
    best2 = np.array([1,0,1])
    for i in range(N+1):
        for j in range(N+1):
            for k in range(N+1):
                f = np.array([i,j,k])*1./N
                (TP,FP,FN) = gnrlBayesCl.binaryMetrics(eta,mu,f)
                print i,j,k
                (loss,g1,g2,g3)=metrics.HMean(TP,FP,FN)
                gradient = ((g1-g2-g3)*eta+g2)*mu
                print np.dot(gradient,best1-f)
                print np.dot(gradient,best2-f)
                print '\n'