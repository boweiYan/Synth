import math
import random
import numpy as np
import numpy.random as rn
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import metrics
import gnrlBayesCl
import optHMean


def array2str(nparray):
    str1=''
    for i in range(nparray.shape[0]):
        str1 += ' '+str(nparray[i])
    return str1

def Hmean_partial_der(f):
    x = f[0]
    y = f[1]
    z = f[2]
    l1 = pow(x-z-200, -2)
    l1 *= (6.2475*x*x+x*(-12.495*z-2499)-25*y*y+y*(-50*z-5000)-18.7525*z*z-2501*z-100)
    l2 = (25*x+50*y+25*z)/(x-z-200)
    l3 = pow(x-z-200, -2)
    l3 *= (18.7525*x*x+x*(50*y+12.495*z-2501)+25*y*y+y*(-5000)-6.2475*z*z-2499*z+100)
    return l1,l2,l3

if __name__=='__main__':
    eta0 = np.array([0.49,0.5,0.51])
    mu0 = np.array([0.25,0.5,0.25])

    # Original problem
    print eta0, mu0
    (fFC, fFS, cF) = gnrlBayesCl.bestfF(eta0, mu0, metrics.HMean, 'F')
    print fFC, fFS, cF
    (fTC, fTS, cT) = gnrlBayesCl.bestfF(eta0, mu0, metrics.HMean, 'T')
    print fTC, fTS, cT
    print '\n'

    # Perturbing eta
    for iter in range(10):
        eta = eta0 + np.array([rn.uniform(-.1,.1), rn.uniform(-.1,.1), rn.uniform(-.1,.1)])
        mu = mu0
        print eta, mu
        (fFC, fFS, cF) = gnrlBayesCl.bestfF(eta, mu, metrics.HMean, 'F')
        print fFC, fFS, cF
        (fTC, fTS, cT) = gnrlBayesCl.bestfF(eta, mu, metrics.HMean, 'T')
        print fTC, fTS, cT
        print '\n'
    '''
    # Perturbing mu
    for iter in range(10):
        mu = mu0 + np.array([rn.uniform(-.1,.1), rn.uniform(-.1,.1), rn.uniform(-.1,.1)])
        mu = mu/sum(mu)
        eta = eta0
        print eta, mu
        (fFC, fFS, cF) = gnrlBayesCl.bestfF(eta, mu, metrics.HMean, 'F')
        print fFC, fFS, cF
        (fTC, fTS, cT) = gnrlBayesCl.bestfF(eta, mu, metrics.HMean, 'T')
        print fTC, fTS, cT
        print '\n'

    # Perturbing both
    for iter in range(10):
        mu = mu0 + np.array([rn.uniform(-.1,.1), rn.uniform(-.1,.1), rn.uniform(-.01,.1)])
        mu = mu/sum(mu)
        eta = eta0 + np.array([rn.uniform(-.1,.1), rn.uniform(-.1,.1), rn.uniform(-.1,.1)])
        print eta, mu
        (fFC, fFS, cF) = gnrlBayesCl.bestfF(eta, mu, metrics.HMean, 'F')
        print fFC, fFS, cF
        (fTC, fTS, cT) = gnrlBayesCl.bestfF(eta, mu, metrics.HMean, 'T')
        print fTC, fTS, cT
        print '\n'
    '''