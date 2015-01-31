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
    if np.isnan(x*(1-x-y-z)/((x+y)*(1-x-y))) or (x+y)*(x+y-1)==0:
        return (float('nan'),float('nan'),float('nan'),float('nan'))
    else:
        loss = np.sqrt(x*(1-x-y-z)/((x+y)*(1-x-y)))
        g1num = x*x*(y-z)+(y-1)*y*(2*x+y+z-1)
        g1den = 2*pow((x+y)*(1-x-y),2)*loss
        g1 = g1num/g1den
        g2num = x*z*(1-2*x-2*y)-x*pow(1-x-y,2)
        g2 = g2num/g1den
        g3den = 2*(x+y)*(x+y-1)*loss
        g3 = x/g3den
        return (loss,g1,g2,g3)

def HMean(x,y,z):
    loss = 2/((x+y)/x+(1-x-y)/(1-x-y-z))
    g1num = 2*(x*x*(y-z)+y*(2*x*(y+z-1)+y*y+2*y*(z-1)+z*z-2*z+1))
    g1den = pow(2*x*x+x*(3*y+z-2)+y*(y+z-1),2)
    g1=g1num/g1den
    g2num = 2*x*(x*x+x*(2*y+3*z-2)+y*y+2*y*(z-1)+z*z-2*z+1)
    g2 = -g2num/g1den
    g3num = 2*x*x*(x+y-1)
    g3 = g3num/g1den
    return (loss,g1,g2,g3)

def QMean(x,y,z):
    loss = 1-0.5*(pow(y/(x+y),2)+pow(z/(1-x-y),2))
    g1 = y*y/pow(x+y,3)+z*z/pow(x+y-1,3)
    g2 = g1-y/pow(x+y,2)
    g3 = -z/pow(1-x-y,2)
    return (loss,g1,g2,g3)
