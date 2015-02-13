from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
fig = plt.figure()
ax = fig.gca(projection='3d')
X=np.arange(0,1,0.01)
Y=X.copy()
xx,yy = np.meshgrid(X,Y)
Z=4*xx*yy/(xx+yy)

print Z
ax.surf = ax.plot_surface(xx, yy, Z, rstride=1, cstride=1)

ax.set_xlabel('X')
ax.set_xlim(0, 1)
ax.set_ylabel('Y')
ax.set_ylim(0, 1)
ax.set_zlabel('Z')
ax.set_zlim(0, 2)

plt.show()

