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

def precwrap(f,eta,mu):
    return -metrics.precision(f,eta,mu)

def bestfWA(eta, mu, lossfunc, method):
    k = eta.shape[0]
    fFC = np.zeros(k)
    fFS = -1
    if method == 'F':
        # res = scipy.optimize.minimize(weightedAvg, (1, 1), args=(eta, mu), bounds=((-1, 1), (-1, 1)))
        ini = np.random.uniform(-1,1,k)
        bounds = ((-1,1),)
        for i in range(k-1):
            bounds += ((-1,1),)
        res = scipy.optimize.minimize(precwrap, ini, args=(eta, mu), bounds=bounds)
        fFC = res.x
        fFS = metrics.precision(fFC,eta,mu)

    elif method == 'T':
        f = np.ones(k)

        for i in range(pow(2,k)):


                    f = np.array([i, j, k])*1./N
                    f = 2*f -1
                    loss = metrics.precision(f, eta, mu)
                    if loss > fFS:
                        fFC = f
                        fFS = loss
    return (fFC, fFS)

if __name__=='__main__':
    eta = np.array([0.5, 0.3, 0.9, .01])
    mu = np.array([0.4, 0.35, 0.15, 0.1])
    (fFC, fFS) = bestfWA(eta, mu, metrics.weightedAvg, 'F')
    print fFC, fFS
    (fTC, fTS) = bestfWA(eta, mu, metrics.weightedAvg, 'T')
    print fTC, fTS
    print '\n'
