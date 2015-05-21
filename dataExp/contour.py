import plotly.plotly as py
from plotly.graph_objs import *
import optWA
import numpy as np
import metrics
'''
# Quadratic Loss
size = 100
x = np.linspace(-1, 1, size)
y = np.linspace(-1, 1, size)
z = np.empty((size, size))
for i, xi in enumerate(x):
    for j, yj in enumerate(y):
        z[i][j] = -(xi+2)*(xi+2)-(yj-.5)*(yj-.5)

data = Data([
    Contour(
        z=z,
        x=x,
        y=y,
        colorscale='Hot'
    )
])
layout = Layout(
    annotations=Annotations([
        Annotation(
            x=.5,
            y=-1,
            xref='x',
            yref='y',
            text='Optimal Continuous Solution',
            showarrow=True,
            font=Font(
                family='',
                size=18
            )
        ),
        Annotation(
            x=1,
            y=-1,
            xref='x',
            yref='y',
            text='Optimal Integer Classifier',
            showarrow=True,
            ax=-30,
            ay=-80,
            font=Font(
                family='',
                size=18
            )
        )
    ])
)
fig = Figure(data=data, layout=layout)
plot_url = py.plot(fig)
# plot_url = py.plot(data, filename='simple-contour')
py.image.save_as(fig, 'quadratic_contour.png')
'''

# 2-dimension Weighted Average
size = 100
eta = np.array([.5,.7])
mu = np.array([.4,.6])
x = np.linspace(-1, 1, size)
y = np.linspace(-1, 1, size)
z = np.empty((size, size))
for i, xi in enumerate(x):
    for j, yj in enumerate(y):
        f = np.array(xi,yj)
        z[i][j] = optWA.weightedAvg(f, eta, mu)

data = Data([
    Contour(
        z=z,
        x=x,
        y=y,
        colorscale='Hot'
    )
])
fig = Figure(data=data)
plot_url = py.plot(fig)
py.image.save_as(fig, 'WA_contour.png')
'''
# 2-dimension F-beta measure
# size = 100
# eta = np.array([.62,.7])
# mu = np.array([.4,.6])
# beta = 1
# x = np.linspace(-1, 1, size)
# y = np.linspace(-1, 1, size)
# z = np.empty((size, size))
# for i, xi in enumerate(x):
#     for j, yj in enumerate(y):
#         f = np.array(xi,yj)
#         z[i][j] = metrics.precision(f, eta, mu)
#         print z[i][j]
#
# data = Data([
#     Contour(
#         z=z,
#         x=x,
#         y=y
#     )
# ])
# fig = Figure(data=data)
# plot_url = py.plot(fig)
# py.image.save_as(fig, 'prec_contour.png')
'''