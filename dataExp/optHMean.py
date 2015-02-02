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


if __name__=='__main__':
    eta = np.array([0.49,0.5,0.51])

    for i in range(100):
        eps  = 0.5/100 * i
        mu = np.array([0.5-eps,2*eps,0.5-eps])
        (bestC,bestS)=gnrlBayesCl.best_classifier(eta,mu,2,3,metrics.HMean)
        py1 = np.dot(eta,mu)

        print "mu="+array2str(mu)+" optimal classifier: "+bestC+" p(true)="+str(py1)
        (TP,FP,FN) = gnrlBayesCl.binaryMetrics(eta,mu,bestC)
        print "Optimal Binary Metrics"+array2str(np.array([TP,FP,FN,1-TP-FP-FN]))

        (loss,g1,g2,g3)=metrics.HMean(TP,FP,FN)
        print "g1 "+str(g1)+" g2 "+str(g2)+" g3 "+str(g3)
