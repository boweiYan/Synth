'''
Created on Jan 28, 2015
Check bayes optimal classifier for general metrics

max_f L(f)
L(f) = g(TP, FP, FN)

@author: Bowei Yan
'''

import numpy as np
import numpy.random as rn
import matplotlib.pyplot as plt
import metrics
import scipy
import optHMean

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

def bestfF(eta, mu, lossfunc, method, dim):
    fFC = np.zeros(dim)
    fFS = float("inf")
    start = np.zeros(dim)
    bnd = ()
    for i in range(dim):
        bnd = bnd + ((-1,1),)
    if method == 'F':
        res = scipy.optimize.minimize(lossfunc, list(start), args=(eta, mu), bounds=bnd)
        fFC = res.x
        fFS = lossfunc(fFC, eta, mu)
        print "optimal classifier: "+str(fFC)+" score: "+str(fFS)

    elif method == 'T':
        # Traverse over 2^10 vertices
        for i in range(1,pow(2,dim)):
            binstr = bin(i)
            binlen = len(binstr)-2
            f = -1*np.ones(dim)
            for j in range(binlen):
                if int(binstr[-j-1])==1:
                    f[-j-1] = 1
            loss = lossfunc(f, eta, mu)

            if (not np.isnan(loss)) and loss < fFS:
                fFC = f
                fFS = loss
        print "Best deterministic classifier: "+str(fFC)+" score: "+str(fFS)
    return (fFC, fFS)


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
    ax.plot([index[0], index[-1]], [delta, delta], marker='o', linewidth=2.0, markersize=8.0)
    if coef>0:
        ax.step(index, cbest[order], marker='s', linewidth=3.0, markersize=8.0) #, where='mid'
    elif coef<0:
        ax.step(index, -cbest[order], marker='s', linewidth=3.0, markersize=8.0)

    ax.legend(['$\eta(x)$', r"$\delta^*=%.2f$"%(delta,), r"$\theta^*$"], loc='upper left', fontsize=15)
    ax.set_xlabel('x', fontsize=30, weight='bold')
    ax.set_ylim([-1.1,1.1])

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
    # plotme=True

    dom = 10
    k = 2
    (eta, mu) = genData(dom, k)
    lossfunc = metrics.HMean
    print "Testing "+str(lossfunc.__name__)
    (bestC1, bestS1) = bestfF(eta, mu, lossfunc, 'F', dom)
    (bestC2, bestS2) = bestfF(eta, mu, lossfunc, 'T', dom)
    print np.sum((1+bestC1)*eta*mu/2)
    print np.sum((1-bestC1)*(1-eta)*mu/2)
    print np.sum((1+bestC2)*eta*mu/2)
    print np.sum((1-bestC2)*(1-eta)*mu/2)
    #subplotter(bestC1, eta, coef[0], coef[1])