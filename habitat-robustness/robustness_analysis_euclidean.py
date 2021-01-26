import pandas as pd
import networkx as nx 
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance_matrix
import pylab
import network_functions as net


df = pd.read_csv ("csik.csv")


g = net.build_habitat_network(df[["x", "y"]], disp = 1500, plot = False)

print("AWC: ", round(net.get_awc(df, disp = 1500, a_max=2000), 3))
# print("Diameter: ", nx.diameter(g))
print("Density: ", round(nx.density(g), 3))

# recalculate = False

# x1, y1, VD = net.degree(g, recalculate)

# g = net.build_habitat_network(df[["x", "y"]], disp = 1500, plot = False)
# x2, y2, VB = net.betweenness(g, recalculate)
# g = net.build_habitat_network(df[["x", "y"]], disp = 1500, plot = False)
# x3, y3, VC = net.closeness(g, recalculate)
# g = net.build_habitat_network(df[["x", "y"]], disp = 1500, plot = False)
# x5, y5, VR = net.rand(g)

# pylab.figure(1, dpi = 500)
# #pylab.title("a)", loc = "left", fontsize = 14)
# pylab.xlabel(r"Fraction of nodes removed ($\rho$)")

# pylab.yticks(np.arange(0, 1.2, 0.1))
# pylab.xticks(np.arange(0, 1.2, 0.1))

# pylab.ylabel(r"Fractional size of largest component ($\sigma$)")
# pylab.plot(x1, y1, "b-", alpha = 0.6, linewidth = 2.0)
# pylab.plot(x2, y2, "g-", alpha = 0.6, linewidth = 2.0)
# pylab.plot(x3, y3, "r-", alpha = 0.6, linewidth = 2.0)
# pylab.plot(x5, y5, "k-", alpha = 0.6, linewidth = 2.0)
# pylab.legend((r"Degree ($VD = %4.3f$)" %(VD), 
#                 "Betweenness ($VB = %4.3f$)" %(VB), 
#                 "Closeness ($VC = %4.3f$)" %(VC), 
#                 "Random ($VR = %4.3f$)" %(VR)), 
#                 loc = "upper right", shadow = False)

# pylab.savefig("hab_creation.pdf", format = "pdf")
# pylab.close(1)

# print("VB: ", VB)
# print("VC: ", VC)
# print("VD: ", VD)
# print("VR: ", VR)

# g = net.build_habitat_network(df[["x", "y"]], disp = 1500, plot = False)

# print("Diameter: ", nx.diameter(g))
# print("Density: ", round(nx.density(g), 3))