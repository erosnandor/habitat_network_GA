import pandas as pd
import networkx as nx 
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance_matrix
import pylab
import network_functions as net


fig, ax = pylab.subplots(2,figsize=(4.5,10))

df = pd.read_csv ("csik.csv")


g = net.build_habitat_network(df[["x", "y"]], disp = 1500, plot = False)


recalculate = False

x1, y1, VD = net.degree(g, recalculate)

g = net.build_habitat_network(df[["x", "y"]], disp = 1500, plot = False)
x2, y2, VB = net.betweenness(g, recalculate)
g = net.build_habitat_network(df[["x", "y"]], disp = 1500, plot = False)
x3, y3, VC = net.closeness(g, recalculate)
g = net.build_habitat_network(df[["x", "y"]], disp = 1500, plot = False)
x5, y5, VR = net.rand(g)

# pylab.figure(1, dpi = 500)
ax[0].set_title("a)", loc = "left", fontsize = 14)
ax[0].set_xlabel(r"Fraction of nodes removed ($\rho$)")

ax[0].set_yticks(np.arange(0, 1.2, 0.1))
ax[0].set_xticks(np.arange(0, 1.2, 0.1))

ax[0].set_ylabel(r"Fractional size of largest component ($\sigma$)")

ax[0].plot(x1, y1, "b-", alpha = 0.6, linewidth = 2.0)
ax[0].plot(x2, y2, "g-", alpha = 0.6, linewidth = 2.0)
ax[0].plot(x3, y3, "r-", alpha = 0.6, linewidth = 2.0)
ax[0].plot(x5, y5, "k-", alpha = 0.6, linewidth = 2.0)



ax[0].legend((r"Degree ($VD = %4.3f$)" %(VD), 
                "Betweenness ($VB = %4.3f$)" %(VB), 
                "Closeness ($VC = %4.3f$)" %(VC), 
                "Random ($VR = %4.3f$)" %(VR)), 
                loc = "upper right", shadow = False)



df = pd.read_csv ("hab_improvement.csv")


g = net.build_habitat_network(df[["x", "y"]], disp = 1500, plot = False)

print("Diameter: ", nx.diameter(g))
print("Density: ", round(nx.density(g), 3))

recalculate = False

x1, y1, VD = net.degree(g, recalculate)

g = net.build_habitat_network(df[["x", "y"]], disp = 1500, plot = False)
x2, y2, VB = net.betweenness(g, recalculate)
g = net.build_habitat_network(df[["x", "y"]], disp = 1500, plot = False)
x3, y3, VC = net.closeness(g, recalculate)
g = net.build_habitat_network(df[["x", "y"]], disp = 1500, plot = False)
x5, y5, VR = net.rand(g)

# pylab.figure(1, dpi = 500)
ax[1].set_title("b)", loc = "left", fontsize = 14)
ax[1].set_xlabel(r"Fraction of nodes removed ($\rho$)")

ax[1].set_yticks(np.arange(0, 1.2, 0.1))
ax[1].set_xticks(np.arange(0, 1.2, 0.1))

ax[1].set_ylabel(r"Fractional size of largest component ($\sigma$)")
ax[1].plot(x1, y1, "b-", alpha = 0.6, linewidth = 2.0)
ax[1].plot(x2, y2, "g-", alpha = 0.6, linewidth = 2.0)
ax[1].plot(x3, y3, "r-", alpha = 0.6, linewidth = 2.0)
ax[1].plot(x5, y5, "k-", alpha = 0.6, linewidth = 2.0)
ax[1].legend((r"Degree ($VD = %4.3f$)" %(VD), 
                "Betweenness ($VB = %4.3f$)" %(VB), 
                "Closeness ($VC = %4.3f$)" %(VC), 
                "Random ($VR = %4.3f$)" %(VR)), 
                loc = "upper right", shadow = False)


pylab.savefig("Habitat_improvement.pdf", bbox_inches='tight')
pylab.close(1)

