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
    N=100
    for i in range(N+1):
        for j in range(N+1):
            for k in range(N+1):
                f=np.array([i,j,k])*1./N
                (TP,FP,FN) = gnrlBayesCl.binaryMetrics(eta,mu,f)
                score = Hmean_ex(TP,1-TP-FP-FN)
                if score > best:
                    best = score
                    bestC = f
                    bestTP = TP
                    bestTN = 1-TP-FP-FN
    print best, bestC, bestTP, bestTN
