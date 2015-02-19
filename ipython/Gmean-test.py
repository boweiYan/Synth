# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

import numpy as np
import math

# define functions
TP = lambda f: (f*eta*mu).sum()
TN = lambda f: ((1-f)*(1-eta)*mu).sum()
FP = lambda f: (f*(1-eta)*mu).sum()
FN = lambda f: ((1-f)*eta*mu).sum()
G = lambda f: np.sqrt( (TP(f)*TN(f))/( (TP(f)+FN(f))*(TN(f)+FP(f)) ) )

# define gradient dL/dC
gTP = lambda f: G(f)/(2*TP(f)) - G(f)/(2*(TP(f) + FN(f)))
gTN = lambda f: G(f)/(2*TN(f)) - G(f)/(2*(TN(f) + FP(f)))
gFN = lambda f: -(G(f)**3)*(TN(f) + FP(f))/(2*TP(f)*TN(f))
gFP = lambda f: -(G(f)**3)*(TP(f) + FN(f))/(2*TP(f)*TN(f))

# define constants c1, c2 so grad(f) = c1*eta(x) + c2
c1 = lambda f: gTP(f) + gTN(f) - gFP(f) - gFN(f)
c2 = lambda f: gFP(f) - gTN(f)
# f* = sign(eta - delta)
delta = lambda f: -c2(f)/c1(f)

# <codecell>

# define probabilities
mu = np.array([.25, .5, .25])
eps=0.09
eta = np.array([0.5-eps, 0.5, 0.5+eps])
print eta

# <codecell>

# Find best classifier
K = len(mu)
bestF = np.zeros(K)
bestS = H(bestF)
for ik in range(int(math.pow(2,K))):
    f = np.array([int(x) for x in np.binary_repr(ik, K)], dtype=float)
    score = G(f)
    print f, score
    if score > bestS:
        bestF = f.copy()
        bestS = score

# <codecell>

print "------"
print bestF
print TP(bestF), FP(bestF), FN(bestF), TN(bestF), G(bestF)
print gTP(bestF), gTN(bestF), gFN(bestF), gFP(bestF)
print c1(bestF), c2(bestF), thresh(bestF)
print eta - delta(bestF)

print "------"
fs = np.array([0,0.5,1])
print fs
print TP(fs), FP(fs), FN(fs), TN(fs), G(fs)
print gTP(fs), gTN(fs), gFN(fs), gFP(fs)
print c1(fs), c2(fs), thresh(fs)
print eta - delta(fs)

# <codecell>


