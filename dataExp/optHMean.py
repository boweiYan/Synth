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
    best1 = np.array([-1,1,-1])
    best2 = np.array([1,-1,1])

    grad1 = Hmean_partial_der(best1)
    grad2 = Hmean_partial_der(best2)
    print "%.7f, %.7f, %.7f" % (grad1[0],grad1[1],grad1[2])
    print "%.7f, %.7f, %.7f" % (grad2[0],grad2[1],grad2[2])
    # f = np.array([1,-1,1])
    # print np.dot(best1-f,Hmean_partial_der(best1))

    #
    # eta = np.array([0.49,0.5,0.51])
    # mu = np.array([0.3,0.4,0.3])
    # best = -1
    # bestTP = -1
    # bestTN = -1
    # f = np.zeros(3)
    # #(bestC,bestS)=gnrlBayesCl.best_classifier(eta,mu,2,3,metrics.HMean)
    #
    # # Test all randomized classifiers
    # N = 100
    # for i in range(N+1):
    #     for j in range(N+1):
    #         for k in range(N+1):
    #             f = np.array([i,j,k])*1./N
    #             f = 2*f -1
    #             (TP,FP,FN) = gnrlBayesCl.binaryMetrics(eta,mu,f)
    #             (loss,g1,g2,g3)=metrics.HMean(TP,FP,FN)
    #             if loss>best:
    #                 best=loss
    #                 bestf=f
    #
    # print bestf,best
    #
    # # Test all randomized classifiers
    # N=1
    # for i in range(N+1):
    #     for j in range(N+1):
    #         for k in range(N+1):
    #             f = np.array([i,j,k])*1./N
    #             f = 2*f -1
    #             (TP,FP,FN) = gnrlBayesCl.binaryMetrics(eta,mu,f)
    #             print f
    #             print TP,FP,FN,1-TP-FP-FN
    #             (loss,g1,g2,g3)=metrics.HMean(TP,FP,FN)
    #             print g1,g2,g3,loss
    #             print ((g1-g2-g3)*eta+g2)*mu/2
    #             print (g1-g2-g3)/2,-g2/(g1-g2-g3)
    #             print '\n'
    #
    # # Now let's check the optimal condition
    # best1 = np.array([-1,1,-1])
    # best2 = np.array([1,-1,1])
    # grad1 = np.array([ 0.05688,0.08,0.06312])
    # grad2 = np.array([-0.06312,-0.08,-0.05688])
    # for i in range(N+1):
    #     for j in range(N+1):
    #         for k in range(N+1):
    #             f = np.array([i,j,k])*1./N
    #             f = 2*f -1
    #             (TP,FP,FN) = gnrlBayesCl.binaryMetrics(eta,mu,f)
    #             print f
    #             (loss,g1,g2,g3)=metrics.HMean(TP,FP,FN)
    #             gradient = ((g1-g2-g3)*eta+g2)*mu/2
    #             print np.dot(grad1,best1-f)
    #             print np.dot(grad2,best2-f)
    #             print '\n'