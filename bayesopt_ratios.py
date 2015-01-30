'''
Created on June 5, 2014
Check bayes optimal classifiier for ratio metrics

max_f L(f) 
L(f) = (a0 + a11TP + a10FP + a01FN + a00TN) / (b0 + b11TP + b10FP + b01FN + b00TN)

@author: ook59
'''
import numpy as np
import numpy.random as rn
import itertools as it
logit = lambda z: 1.0/(1.0 + np.exp(-z))

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


def subplotter(cbest, eta, delta, alist, blist):
    ''' plot the classifier vs eta '''
    index = np.arange(len(eta))+1
    order = eta.argsort()
    
    f = plt.figure()
    ax = f.add_subplot(111)
    ax.step(index, eta[order], 'k-',  linewidth=3.0, markersize=8.0)
    # TODO, use lolipop plot!
    ax.plot([index[0], index[-1]], [delta, delta], marker='o', linewidth=3.0, markersize=8.0)
    ax.step(index, cbest[order], marker='s', linewidth=3.0, markersize=8.0) #, where='mid'
    
    scores = ["", "TP", "FP", "FN", "TN"]
    atop = []
    for id, (a, sc) in enumerate(zip(alist, scores)):
        if a==0:
            continue
        if a==1 and id>0:
            atop.append("%s"%(sc,))
            continue
        atop.append("%d%s"%(a, sc,))
    
    afirst = "+".join(s for s in atop)
    
    atop = []
    for id, (a, sc) in enumerate(zip(blist, scores)):
        if a==0:
            continue
        if a==1 and id>0:
            atop.append("%s"%(sc,))
            continue
        atop.append("%d%s"%(a, sc,))
    asecond = "+".join(s for s in atop)
    bayesprint = r'$\frac{%s}{%s}$'%(afirst, asecond)
    
    ax.legend(['$\eta(x)$', r"$\delta^*=%.2f$"%(delta,), r"$\theta^*$"], loc='upper left', fontsize=35)
    ax.set_xlabel('x', fontsize=30, weight='bold')
    #plt.ylabel('f(x)', fontsize=20)#, weight='bold')
    ax.set_ylim([-.1,1.1])
    
    left, width = .25, .745
    bottom, height = .015, .5
    right = left + width
    top = bottom + height
    ax.text(right, bottom, bayesprint, horizontalalignment='right',verticalalignment='bottom', transform=ax.transAxes, fontsize=40, weight='bold')
    #ax.set_title(bayesprint, fontsize=40, y=1.03)
    plt.tight_layout()
    ApplyFont(plt.gca())
    if plotme:
        plt.show()
    else:
        filename = "num_"+afirst+"_"+"den_"+asecond
        plt.savefig(filename+".pdf")
        plt.clf()
        plt.close("all")

##########################################
# Population Measures
##########################################
def TP(eta, mu, f):
    ''' P(f = 1,  y=1)
    eta = P(Y=1|x)
    mu = P(x)
    f = classifier decisions (binary)
    '''
    tp = np.dot(eta*f, mu)
    return tp

def PHAT(eta, mu, f):
    ''' P(f = 1)
    eta = P(Y=1|x)
    mu = P(x)
    f = classiifier decisions (binary)
    '''
    phat = np.dot(f, mu)
    return phat

##########################################
# classifier
##########################################
from operator import mul
def costfunc(f, eta, mu, consts):
    ''' computes generalized men cost function
    args = (eta, mu, f, consts)
    consts = c0, c1, c2, d0, d1, d2
    '''
    c0, c1, c2, d0, d1, d2 = consts
    tp = TP(eta, mu, f)
    phat = PHAT(eta, mu, f)
    num = c0 + c1*tp + c2*phat
    den = d0 + d1*tp + d2*phat
    thecost = num/den
    return thecost

def best_classifier(eta, mu, consts):
    ''' Tests for best classiifier by enumeratuive search
    eta = P(Y=1|x)
    mu = P(x)
    consts = c0, c1, c2, d0, d1, d2
    '''
    bestS = -1.0
    K = len(mu)
    # convert K length binary variable to binary
    for ik in it.count():
        classif = np.array([int(x) for x in np.binary_repr(ik, K)])
        if len(classif) > K:
            break
        fclass = costfunc(classif, eta, mu, consts)
        if fclass > bestS:
            bestC = classif.copy()
            bestS = fclass
        #print classif[eta.argsort()], fclass

    return  bestC, bestS
    
def test_best_classifier(alist, blist, K=10, D=1):
    ''' generate data and test best (generalized mean) classifier
    a0, a11, a10, a01, a00 = alist
    b0, b11, b10, b01, b00 = blist
    K = size of support
    D = dimension of X's (def = 1)
    '''
    a0, a11, a10, a01, a00 = alist
    b0, b11, b10, b01, b00 = blist
    # Generate data
    xlib  = rn.randn(K, D) # dictionary for possible values for x
    # set p(X)
    mu  = np.abs(np.random.randn(K)) # p(X)
    mu /= mu.sum()
    # Model
    wtrue = rn.randn(D)
    g = np.dot(xlib, wtrue)
    eta = logit(g) # P(Y|X)
    pi = np.dot(eta, mu)
    
    ###################################
    # fix constants
    ###################################
    c0 = a01*pi + a00 - a00*pi + a0
    c1 = float(a11 - a10 - a01 + a00)
    c2 = float(a10 - a00)
    d0 = float(b01*pi + b00 - b00*pi + b0)
    d1 = float(b11 - b10 - b01 + b00)
    d2 = float(b10 - b00)
    
    consts = c0, c1, c2, d0, d1, d2
    ###################################
    # 1. Optimal classifier
    ###################################
    cbest, sbest = best_classifier(eta, mu, consts)
    
    ###################################
    # Threshold classifier
    ###################################
    delta = (d2*sbest - c2) / (c1 - d1*sbest)
    #print delta
    
    if np.abs(delta) == -0.0: delta = 0.0
    
    ###################################
    # plot results
    ###################################
    print "c0=%g, c1=%g, c2=%g, d0=%g, d1=%g, d2=%g"%consts
    print "L^*=%g, thresh=%g"%(sbest, delta,)

    subplotter(cbest, eta, delta, alist, blist)
    
    #return cbest, eta, sbest
    
def main():
    a0  = 1
    a11 = 0
    a10 = 0
    a01 = 0
    a00 = 0
    
    b0  = 0
    b11 = 1
    b10 = 0
    b01 = 0
    b00 = 1
    
    alist = np.array([a0, a11, a10, a01, a00], dtype=float)
    blist = np.array([b0, b11, b10, b01, b00], dtype=float)
    
    test_best_classifier(alist, blist, K=10, D=1)

if __name__ == '__main__':
    main()
    
    