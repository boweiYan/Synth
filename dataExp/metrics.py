###################
# Define various cost functions:
# Input: TP,FP,FN
# Output: score, g1, g2, g3
###################

import math
import numpy as np
import numpy.random as rn
import random

def precision(f,eta,mu):
    TP = np.sum((1+f)*eta*mu/2)
    FN = np.sum((1-f)*eta*mu/2)
    TN = np.sum((1-f)*(1-eta)*mu/2)
    FP = np.sum((1+f)*(1-eta)*mu/2)
    if TP+FP==0:
        loss = -1
    else:
        loss = TP/(TP+FP)
    return loss

def Fbeta(f,eta,mu,beta):
    TP = np.sum((1+f)*eta*mu/2)
    FN = np.sum((1-f)*eta*mu/2)
    TN = np.sum((1-f)*(1-eta)*mu/2)
    FP = np.sum((1+f)*(1-eta)*mu/2)
    loss = (1+beta*beta)*TP/((1+beta*beta)*TP+FP+beta*beta*FN)
    return loss

def weightedAvg(f, eta, mu):
    TP = np.sum((1+f)*eta*mu/2)
    # FN = np.sum((1-f)*eta*mu/2)
    TN = np.sum((1-f)*(1-eta)*mu/2)
    # FP = np.sum((1+f)*(1-eta)*mu/2)

    loss = -(2*TP+2*TN)/(1+TP+TN)
    return loss

def Jaccard(x,y,z):
    loss = x/(x+y+z)
    g1 = (y+z)/pow(x+y+z,2)
    g2 = -x/pow(x+y+z,2)
    g3 = g2
    return (loss,g1,g2,g3)

def GMean(x,y,z):
    loss = np.sqrt(x*(1-x-y-z)/((x+z)*(1-x-z)))
    g1num = x*x*(y-z)+(z-1)*z*(1-2*x-y-z)
    g1den = 2*pow(x+z-1,2)*pow(x+z,2)*loss
    g1 = -g1num/g1den
    g2num = x
    g2den = 2*(x+z-1)*(x+z)*loss
    g2 = g2num/g2den
    g3num = x*(x*x+2*x*(y+z-1)+y*(2*z-1)+pow(z-1,2))
    g3 = -g3num/g1den
    return (loss,g1,g2,g3)

def HMean(f, eta, mu):
    TP = np.sum((1+f)*eta*mu/2)
    FN = np.sum((1-f)*eta*mu/2)
    TN = np.sum((1-f)*(1-eta)*mu/2)
    FP = np.sum((1+f)*(1-eta)*mu/2)

    loss = -2/((TP+FN)/TP+(FP+TN)/TN)
    return loss

def HMean_coef(f, eta, mu):
    TP = np.sum((1+f)*eta*mu/2)
    FN = np.sum((1-f)*eta*mu/2)
    TN = np.sum((1-f)*(1-eta)*mu/2)
    FP = np.sum((1+f)*(1-eta)*mu/2)

    p = TP+FN
    g1 = 2*p*TN*TN/pow(TP+p*TN-p*TP,2)
    g2 = 2*(1-p)*TP*TP/pow(TP+p*TN-p*TP,2)
    c1 = .5*(g1+g2)
    c2 = g2/(g1+g2)
    return c1,c2

def test1(f, eta, mu):
    TP = np.sum((1+f)*eta*mu/2)
    FN = np.sum((1-f)*eta*mu/2)
    TN = np.sum((1-f)*(1-eta)*mu/2)
    FP = np.sum((1+f)*(1-eta)*mu/2)

    loss = pow(TP-TN,2)
    return loss

def QMean(x,y,z):
    loss = 1-0.5*(pow(z/(x+z),2)+pow(y/(1-x-z),2))
    g1 = y*y/pow(x+z-1,3)+z*z/pow(x+z,3)
    g2 = -y/pow(x+z-1,2)
    g3 = y*y/pow(x+z-1,3)-x*z/pow(x+z,3)
    return (loss,g1,g2,g3)

def TPR(x,y,z):
    loss=x/(x+z)
    g1 = x/pow(x+z,2)
    g2 = 0
    g3 = -x/(x+z)/(x+z)
    return (loss,g1,g2,g3)