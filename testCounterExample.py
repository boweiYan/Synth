'''
Created on Jan 28, 2015
Check bayes optimal classifier for general metrics

max_f L(f)
L(f) = g(TP, FP, FN)

@author: Bowei
'''
import math
import numpy as np
import numpy.random as rn
import random
import matplotlib.pyplot as plt
logit = lambda z: 1.0/(1.0 + np.exp(-z))

def genData(dom,k):
    '''Input:
    dom: domain of x
    k: number of classes
    '''
# Generate data
    D=1
    xlib  = rn.randn(dom, D) # dictionary for possible values for x
    # set p(X)
    mu  = np.abs(np.random.randn(dom)) # p(X)
    mu /= mu.sum()
    # Model
    wtrue = rn.randn(D)
    g = np.dot(xlib, wtrue)
    eta = logit(g) # P(Y|X)
    print "True prob: " + str(eta)
    print "True dist: " + str(mu)

    return (eta,mu)

def binary2nparray(fstr):
    f=np.zeros(len(fstr))
    for i in range(len(fstr)):
        f[i]=int(fstr[i])
    return f

def binaryMetrics(eta,mu,f_str):
    f=binary2nparray(f_str)
    TP = np.dot(eta*f, mu)
    FP = np.dot((1-eta)*f, mu)
    FN = np.dot(eta*(1-f),mu)
    return (TP,FP,FN)

def binaryMetrics_emp(ypred,ytrue):
    N=ytrue.shape[0]
    TP=np.sum((ypred==1)*(ytrue==1))/float(N)
    FP=np.sum((ypred==1)*(ytrue==0))/float(N)
    FN=np.sum((ypred==0)*(ytrue==1))/float(N)
    return (TP,FP,FN)

def oracleClassifier(fopt,lossfunc,eta,mu):
    dom = eta.shape[0]
    (TP,FP,FN) = binaryMetrics(eta,mu,fopt)
    print "Optimal Binary Metrics"
    print (TP,FP,FN,1-TP-FP-FN)

    (loss,g1,g2,g3)=lossfunc(TP,FP,FN)
    coef = g1-g2-g3
    thres = -g2/coef
    print "g1 "+str(g1)+" g2 "+str(g2)+" g3 "+str(g3)
    print "Our classifier= "+str(coef)+"*sgn(eta(x)-"+str(thres)+")"
    f=''
    for i in range(dom):
        f += str(int(coef*(eta[i]-thres)>=0))
    (TP,FP,FN) = binaryMetrics(eta,mu,f)
    #print "Binary Metrics"
    #print (TP,FP,FN)
    score=lossfunc(TP,FP,FN)[0]
    return f,score,coef,thres

def best_classifier(eta,mu,k,dom,lossfunc):
    bestC = np.zeros(dom)
    curC = bestC
    bestS = -1
    # convert K length binary variable to binary
    for i in range(int(math.pow(k,dom))):
        curC = np.binary_repr(i,dom)
        (TP,FP,FN)=binaryMetrics(eta,mu,curC)
        curS=lossfunc(TP,FP,FN)[0]
        if ~np.isnan(curS) and curS > bestS:
            bestC = curC
            bestS = curS

    return (bestC,bestS)

###################
# Define various cost functions:
# Input: TP,FP,FN
# Output: score, g1, g2, g3
###################
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
    loss = np.sqrt(x*(1-x-y-z)/((x+y)*(1-x-y)))
    g1num = x*x*(y-z)+(y-1)*y*(2*x+y+z-1)
    g1den = 2*pow((x+y)*(1-x-y),2)*loss
    g1 = g1num/g1den
    g2num = x*z*(1-2*x-2*y)-x*pow(1-x-y,2)
    g2 = g2num/g1den
    g3 = x/g1den
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

##########################################
# Plotting
##########################################
plotme = True
if not plotme:
    import matplotlib
    matplotlib.use('Agg')
import matplotlib.pyplot as plt

def SetPlotRC():
    #If fonttype = 1 doesn't work with LaTeX, try fonttype 42.
    plt.rc('pdf',fonttype = 1)
    plt.rc('ps',fonttype = 1)

def ApplyFont(ax):

    ticks = ax.get_xticklabels() + ax.get_yticklabels()

    text_size = 14.0

    for t in ticks:
        t.set_fontname('Times New Roman')
        t.set_fontsize(text_size)

    txt = ax.get_xlabel()
    txt_obj = ax.set_xlabel(txt)
    txt_obj.set_fontname('Times New Roman')
    txt_obj.set_fontsize(text_size)

    txt = ax.get_ylabel()
    txt_obj = ax.set_ylabel(txt)
    txt_obj.set_fontname('Times New Roman')
    txt_obj.set_fontsize(text_size)

    txt = ax.get_title()
    txt_obj = ax.set_title(txt)
    txt_obj.set_fontname('Times New Roman')
    txt_obj.set_fontsize(text_size)

def subplotter(cbest_str, eta, coef, delta):
    ''' plot the classifier vs eta '''
    index = np.arange(len(eta))+1
    order = eta.argsort()
    cbest = binary2nparray(cbest_str)
    f = plt.figure()
    ax = f.add_subplot(111)
    ax.step(index, eta[order], 'k-',  linewidth=3.0, markersize=8.0)
    ax.plot([index[0], index[-1]], [delta, delta], marker='o', linewidth=3.0, markersize=8.0)
    if coef>0:
        ax.step(index, cbest[order], marker='s', linewidth=3.0, markersize=8.0) #, where='mid'
    elif coef<0:
        ax.step(index, -cbest[order], marker='s', linewidth=3.0, markersize=8.0)

    ax.legend(['$\eta(x)$', r"$\delta^*=%.2f$"%(delta,), r"$\theta^*$"], loc='upper left', fontsize=35)
    ax.set_xlabel('x', fontsize=30, weight='bold')
    ax.set_ylim([-.1,1.1])

    left, width = .25, .745
    bottom, height = .015, .5
    right = left + width
    top = bottom + height
    ax.text(right, bottom, str(lossfunc.__name__), horizontalalignment='right',verticalalignment='bottom', transform=ax.transAxes, fontsize=40, weight='bold')

    plt.tight_layout()
    ApplyFont(plt.gca())
    if plotme:
        plt.show()
    else:
        filename = "num_"+afirst+"_"+"den_"+asecond
        plt.savefig(filename+".pdf")
        plt.clf()
        plt.close("all")

if __name__=='__main__':
    ##########################
    # Set the parameters HERE!
    ##########################
    dom = 3
    k = 2
    plotme=True
    #(eta,mu)=genData(dom,k)
    eta = np.array([0.49,0.5,0.51])
    mu = np.array([0.25,0.5,0.25])
    lossfunc = QMean
    print "Testing "+str(lossfunc.__name__)
    (bestC,bestS)=best_classifier(eta,mu,k,dom,lossfunc)
    print "optimal classifier"
    print bestC,bestS

    (f, score, coef, thres)=oracleClassifier(bestC,lossfunc,eta,mu)
    print "our classifier"
    print f,score
    if score==bestS:
        print "Success!"
    else:
        "Not Optimal!"
    if np.isnan(score):
        print "Not applicable"
    else:
        subplotter(bestC, eta, coef, thres)