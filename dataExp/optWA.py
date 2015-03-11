import math
import random
import numpy as np
import numpy.random as rn
import scipy.optimize
import metrics
import gnrlBayesCl


def array2str(nparray):
    str1=''
    for i in range(nparray.shape[0]):
        str1 += ' '+str(nparray[i])
    return str1

def weightedAvg(f, eta, mu):
    TP = np.sum((1+f)*eta*mu/2)
    # FN = np.sum((1-f)*eta*mu/2)
    TN = np.sum((1-f)*(1-eta)*mu/2)
    # FP = np.sum((1+f)*(1-eta)*mu/2)

    loss = -(2*TP+2*TN)/(1+TP+TN)
    return loss

def bestfWA(eta, mu, lossfunc, method):
    fFC = np.zeros(3)
    fFS = -1
    if method == 'F':
        # res = scipy.optimize.minimize(weightedAvg, (1, 1), args=(eta, mu), bounds=((-1, 1), (-1, 1)))
        res = scipy.optimize.minimize(weightedAvg, (.37, .28, -.11), args=(eta, mu), bounds=((-1, 1), (-1, 1), (-1, 1)))
        fFC = res.x
        (TP,FP,FN) = gnrlBayesCl.binaryMetrics(eta,mu,fFC)
        print TP+FN
        (fFS,g1,g2,g3) = lossfunc(TP,FP,FN)
        grad = ((g1-g2-g3)*eta+g2)*mu/2
        c1 = (g1-g2-g3)/2
        c2 = -g2/(g1-g2-g3)
        c = (c1, c2, grad)

    elif method == 'T':
        N = 1
        for i in range(N+1):
            for j in range(N+1):
                for k in range(N+1):
                    f = np.array([i, j, k])*1./N
                    f = 2*f -1
                    (TP,FP,FN) = gnrlBayesCl.binaryMetrics(eta,mu,f)
                    (loss,g1,g2,g3) = lossfunc(TP,FP,FN)
                    grad = ((g1-g2-g3)*eta+g2)*mu/2
                    c1 = (g1-g2-g3)/2
                    c2 = -g2/(g1-g2-g3)
                    if loss > fFS:
                        fFC = f
                        fFS = loss
                        c = (c1, c2, grad)
    return (fFC, fFS, c)

if __name__=='__main__':
    eta = np.array([0.5, 0.5, 0.5])
    mu = np.array([0.4, 0.35, 0.25])
    (fFC, fFS, cF) = bestfWA(eta, mu, metrics.weightedAvg, 'F')
    print fFC, fFS, cF
    (fTC, fTS, cT) = bestfWA(eta, mu, metrics.weightedAvg, 'T')
    print fTC, fTS, cT
    print '\n'
