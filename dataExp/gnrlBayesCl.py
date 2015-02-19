'''
Created on Jan 28, 2015
Check bayes optimal classifier for general metrics

max_f L(f)
L(f) = g(TP, FP, FN)

@author: Bowei
'''
import math
import random
import numpy as np
import numpy.random as rn
import matplotlib.pyplot as plt
import metrics

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

def binaryMetrics(eta,mu,f):
    TP = np.dot(eta*(f+1)/2., mu)
    FP = np.dot((1-eta)*(f+1)/2., mu)
    FN = np.dot(eta*(1-f)/2.,mu)
    return (TP,FP,FN)

def binaryMetrics_emp(ypred,ytrue):
    N=ytrue.shape[0]
    TP=np.sum((ypred==1)*(ytrue==1))/float(N)
    FP=np.sum((ypred==1)*(ytrue==0))/float(N)
    FN=np.sum((ypred==0)*(ytrue==1))/float(N)
    return (TP,FP,FN)

def oracleClassifier(fopt,lossfunc,eta,mu):
    dom = eta.shape[0]
    fopt = binary2nparray(fopt)
    (TP,FP,FN) = binaryMetrics(eta,mu,fopt)

    (loss,g1,g2,g3)=lossfunc(TP,FP,FN)
    coef = g1-g2-g3
    thres = -g2/coef
    print "Our classifier= "+str(coef)+"*sgn(eta(x)-"+str(thres)+")"
    f=''
    for i in range(dom):
        f += str(int(coef*(eta[i]-thres)>=0))
    f_arr = binary2nparray(f)
    (TP,FP,FN) = binaryMetrics(eta,mu,f_arr)
    #print "Binary Metrics"
    #print (TP,FP,FN)
    score=lossfunc(TP,FP,FN)[0]
    return f,score,coef,thres

def bestfF(eta, mu, lossfunc, method):
    if method == 'F':
        N = 100
    elif method == 'T':
        N = 1
    fFC = np.zeros(3)
    fFS = -1
    for i in range(N+1):
        for j in range(N+1):
            for k in range(N+1):
                f = np.array([i,j,k])*1./N
                f = 2*f -1
                (TP,FP,FN) = binaryMetrics(eta,mu,f)
                (loss,g1,g2,g3) = lossfunc(TP,FP,FN)
                grad = ((g1-g2-g3)*eta+g2)*mu/2
                c1 = (g1-g2-g3)/2
                c2 = -g2/(g1-g2-g3)
                if loss > fFS:
                    fFC = f
                    fFS = loss
                    c = (c1,c2,grad)
    return (fFC, fFS, c)


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
    plotme=True

    dom = 10
    k = 2
    (eta,mu)=genData(dom,k)
    '''
    # The counter example is here!
    dom = 3
    k = 2
    eta = np.array([0.49,0.5,0.51])
    mu = np.array([0.25,0.5,0.25])
    '''
    lossfunc = metrics.HMean
    print "Testing "+str(lossfunc.__name__)
    (bestC,bestS)=best_classifier(eta,mu,k,dom,lossfunc)
    print "optimal classifier: "+bestC+" score: "+str(bestS)

    (f, score, coef, thres)=oracleClassifier(bestC,lossfunc,eta,mu)
    print "our classifier: "+f+" score: "+str(score)
    if score==bestS:
        print "Success!"
    else:
        "Not Optimal!"
    if np.isnan(score):
        print "Not applicable"
    else:
        subplotter(bestC, eta, coef, thres)