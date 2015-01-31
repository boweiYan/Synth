###################
# Define various cost functions:
# Input: TP,FP,FN
# Output: score, g1, g2, g3
###################

import math
import numpy as np
import numpy.random as rn
import random

def precision(TP,FP,FN):
    loss=TP/(TP+FP)
    g1=FP/(TP+FP)/(TP+FP)
    g2=-TP/(TP+FP)/(TP+FP)
    g3=0
    return (loss,g1,g2,g3)

def F1(TP,FP,FN):
    loss=2*TP/(2*TP+FP+FN)
    g1=(2*FP+2*FN)/(2*TP+FP+FN)/(2*TP+FP+FN)
    g2=-2*TP/(2*TP+FP+FN)/(2*TP+FP+FN)
    g3=g2
    return (loss,g1,g2,g3)

def weightedAvg(x,y,z):
    loss = (2-2*y-2*z)/(2-y-z)
    g1 = 0
    g2 = -2/pow(2-y-z,2)
    g3 = g2
    return (loss,g1,g2,g3)

def Jaccard(x,y,z):
    loss = x/(x+y+z)
    g1 = (y+z)/pow(x+y+z,2)
    g2 = -x/pow(x+y+z,2)
    g3 = g2
    return (loss,g1,g2,g3)

def GMean(x,y,z):
    loss = np.sqrt(x*(1-x-y-z)/((y+z)*(1-y-z)))
    g1num = 2*x+y+z-1
    g1den = 2*(y+z-1)*(y+z)*loss
    g1 = g1num/g1den
    g2num = x*(x*(2*y+2*z-1)+pow(1-y-z,2))
    g2den = 2*pow(1-y-z,2)*pow(y+z,2)*loss
    g2 = -g2num/g2den
    g3 = g2
    return (loss,g1,g2,g3)

def HMean(x,y,z):
    if x==0 or 1-x-z==0:
        return (float('nan'),float('nan'),float('nan'),float('nan'))
    else:
        loss = 2/((x+z)/x+(1-x-z)/(1-x-y-z))
        g1num = 2*(x*x*(z-y)+2*x*z*(y+z-1)+z*pow(1-y-z,2))
        g1den = pow(2*x*x+x*(y+3*z-2)+z*(y+z-1),2)
        g1 = g1num/g1den
        g2num = 2*x*x*(x+z-1)
        g2 = g2num/g1den
        g3num = 2*x*(x*x+x*(3*y+2*z-2)+pow(1-y-z,2))
        g3 = -g3num/g1den
        return (loss,g1,g2,g3)

def QMean(x,y,z):
    loss = 1-0.5*(pow(z/(x+z),2)+pow(y/(1-x-z),2))
    g1 = y*y/pow(x+z-1,3)+z*z/pow(x+z,3)
    g2 = -y/pow(x+z-1,2)
    g3 = y*y/pow(x+z-1,3)-x*z/pow(x+z,3)
    return (loss,g1,g2,g3)
