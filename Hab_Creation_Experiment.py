from src import HabitatCreation as GA
from src import NetworkRobustness as NR

import matplotlib.pyplot as plt

import pandas as pd
import numpy as np
import networkx as nx 
from scipy.spatial import distance_matrix


# ======= SPECIES AND HABITAT RELATED PARAMETERS ======= 

# THE SPATIAL EXTENT OF THE AREA: X0, Y0, X1, Y1

X0 = 560227.9547999999485910
Y0 = 544842.1668000000063330

X1 = 564087.9547999999485910
Y1 = 547602.1668000000063330

MIN_CHROMOSOME_LENGTH = 10 # MINIMUM NUMBER OF HABITATS/PONDS
MAX_CHROMOSOME_LENGTH = 80 # MAXIMUM NUMBER OF HABITATS/PONDS

A_MIN = 1000 # MINIMUM POND AREA IN SQUARE METERS
A_MAX = 2000 # MAXIMUM POND AREA IN SQ. METERS

DISP = 1500 # DISPERSAL CAPABILITY OF THE SPECIES (EXPRESSED IN METERS)

# ======= PARAMETERS OF THE GENETIC ALGORITHM =======

MIN_DISTANCE = 150 # THE MINIMUM DISTANCE BETWEEN PONDS AT THE INITIALIZATION STEP

POPULATION_SIZE = 100 # THE NR. OF INDIVIDUALS IN THE INITIAL POPULATION.

P_INSERT = 0.1 # THRESHOLD FOR POND INSERTION
P_RELOC = 0.8 # THRESHOLD FOR POND RELOCATION

PROP_ELIT = 0.2 # PROPORTION OF ELITES
NR_OFFSPRINGS = 100 

STOP_STATMENT = 1e-6
MAX_GENERATIONS = 150 # I.E., THE SECOND STOP STATEMENT

SEED = 4 # RANDOMIZATION SEED

a = -0.2 # SKEWNESS OF THE LOGISTIC CURVE ==> SEE THE ARTICLE FOR THIS
ALPHA_VECT = [0.1, 0.5, 0.7] # THE VECTOR OF ALPHA (THEORETICAL OPTIMUM) VALUES ==> SEE THE ARTICLE FOR THESE
OMEGA_VECT = [0.25,0.25,0.25,0.25] # THE WEIGHT OF ADDENDS IN THE FITNESS FUNCTION ==> SEE THE ARTICLE FOR THESE


T = 5000 # ITERATION NUMBER FOR HABITAT NETWORK ROBUSTNESS ANALYSIS ==> THE ANALYSIS WILL BE IMPLEMENTED BOTH FOR ORIGINAL AND IMPROVED HABITATS
        # THEREFORE, IT GIVES 10 000 ITERATIONS

# ======= THE GENETIC ALGORITHM =======

indiv, result = GA.genetic_algorithm(X0, Y0, X1, Y1, MIN_CHROMOSOME_LENGTH, MAX_CHROMOSOME_LENGTH, a, A_MIN, A_MAX, 
                                    DISP, MIN_DISTANCE, POPULATION_SIZE, P_INSERT,P_RELOC,PROP_ELIT, NR_OFFSPRINGS, STOP_STATMENT, 
                                    MAX_GENERATIONS, SEED, ALPHA_VECT, OMEGA_VECT)


# ======= SAVING THE ALGORITHM OUTPUTS =======
# Outputs will be saved in the Habitat Creation Outputs directory in CSV format. 

# The "Spatial Output Habitat Creation.csv" contains the coordinates, area and radius of generated habitat patches. 
indiv.to_csv("Habitat Creation Outputs/Spatial Output Habitat Creation.csv")

# "Model Parameters Habitat Creation.csv" contains the model parameters and the alteration of fitness values generation-by-generation. 
result.to_csv("Habitat Creation Outputs/Model Parameters Habitat Creation.csv")



# ======= PLOTTING AND SAVING THE "CONVERGENCE" OF THE FITNESS =======
# The plot will be saved in the Habitat Improvement Outputs directory in PDF format. 

plt.plot(result.Generation_number, result.Fitness)
plt.xlabel("Generation number", fontsize = 11)
plt.ylabel ("Fitness value", fontsize = 11)

plt.grid()
plt.savefig("Habitat Creation Outputs/Fitness Habitat Creation.pdf", bbox_inches='tight')
plt.close(1)

# ======= HABITAT NETWORK ROBUSTNESS: VALIDATION =======

"""
HABITAT ROBUSTNESS ANALYSIS
""" 
print("Robustness anaylsis ... ")
recalculate = False # Recalculation of centrality scores

results = []
for i in range(T):
    graph = NR.build_habitat_network(indiv[["x", "y"]], disp = DISP, plot = False)
    results.append(NR.rand(graph, i))

results = np.concatenate(results)
results = pd.DataFrame(results, columns=["Node_fraction","Largest_comp_fraction"])
results = results.groupby("Node_fraction").agg(Min=('Largest_comp_fraction', 'min'), Mean=('Largest_comp_fraction', np.mean), Max=('Largest_comp_fraction', 'max'))

g = NR.build_habitat_network(indiv[["x", "y"]], disp = DISP, plot = False)
x1,y1,VD = NR.degree(g, recalculate)

g = NR.build_habitat_network(indiv[["x", "y"]], disp = DISP, plot = False)
x2, y2, VB = NR.betweenness(g, recalculate)

plt.plot(results.index, results.Min, lw = 1, color = "dimgray")
plt.plot(results.index, results.Max, lw = 1, color = "dimgray")
plt.fill_between(results.index, results.Min, results.Max, facecolor='gray', alpha=0.5, label='Full range of random failure')
plt.plot(results.index,results.Mean, lw=1.5, label='Mean of random failure', color='black')
plt.plot(x1,y1, lw = 1.5, color = "firebrick", label = "Degree attack")
plt.plot(x2,y2, lw = 1.5, color = "darkgreen", label = "Betweenness attack")


plt.legend(loc='upper right')

plt.grid()

plt.xlabel(r"Fraction of nodes removed ($\rho$)")

plt.yticks(np.arange(0, 1.2, 0.1))
plt.xticks(np.arange(0, 1.2, 0.1))

plt.ylabel(r"Fractional size of largest component ($\sigma$)")

plt.savefig("Habitat Creation Outputs/Habitat Creation Robustness.pdf", bbox_inches='tight')
plt.close(1)