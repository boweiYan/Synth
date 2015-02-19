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
H = lambda f: (2*TP(f)*TN(f))/(2*TP(f)*TN(f) + FP(f)*TP(f) + FN(f)*TN(f) )

# define gradient
hTP = lambda f: (2*FN(f)*(TN(f)**2))/((TN(f)*FN(f) + TP(f)*(2*TN(f) + FP(f)) )**2)
hTN = lambda f: (2*FP(f)*(TP(f)**2))/((TP(f)*FP(f) + TN(f)*(2*TP(f) + FN(f)) )**2)
hFN = lambda f: (-2*TP(f)*(TN(f)**2))/((TN(f)*FN(f) + 2*TP(f)*TN(f) + FP(f)*TP(f) )**2)
hFP = lambda f: (-2*TN(f)*(TP(f)**2))/((TP(f)*FP(f) + 2*TP(f)*TN(f) + FN(f)*TN(f) )**2)

# define constants c1, c2 so grad(f) = c1*eta(x) + c2
c1 = lambda f: hTP(f) + hTN(f) - hFP(f) - hFN(f)
c2 = lambda f: hFP(f) - hTN(f)
# f* = sign(eta - delta)
delta = lambda f: -c2(f)/c1(f)

# <codecell>

# define probabilities
mu = np.array([.25, .5, .25])
eps=0.3
eta = np.array([0.5-eps, 0.5, 0.5+eps])
print eta

# <codecell>

# Find best classifier
K = len(mu)
bestF = np.zeros(K)
bestS = H(bestF)
for ik in range(int(math.pow(2,K))):
    f = np.array([int(x) for x in np.binary_repr(ik, K)], dtype=float)
    score = H(f)
    print f, hTP(f), hTN(f), hFN(f), hFP(f), score
    if score > bestS:
        bestF = f.copy()
        bestS = score

# <codecell>

print "------"
print bestF
print TP(bestF), FP(bestF), FN(bestF), TN(bestF), H(bestF)
print hTP(bestF), hTN(bestF), hFN(bestF), hFP(bestF)
print c1(bestF), c2(bestF), thresh(bestF)
print eta - delta(bestF)

print "------"
fs = np.array([0,0.5,1])
print fs
print TP(fs), FP(fs), FN(fs), TN(fs), H(fs)
print hTP(fs), hTN(fs), hFN(fs), hFP(fs)
print c1(fs), c2(fs), thresh(fs)
print eta - delta(fs)

# <codecell>



# <codecell>


# <codecell>


# <codecell>


