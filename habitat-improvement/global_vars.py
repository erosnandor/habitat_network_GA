# The aim of this Genetic Algorithm to generate optimal amphibian habitats within a spatial extent. 
# The spatial extent in our case was given in Stereo70 / Dealurile Piscului projection system from Romania. The usage of local, Euclidean projection system is important, because the distances between habitats are measured in meters. 

# We run this algorithm as a case study for the moor frog (Rana arvalis), which is one of the rarest amphibian species in Romania. 

# @Authors: Bajkó Péter, Botos Barbara, Erős Nándor, Kovács Levente

# Author for correspondence: Erős Nándor, e-mail: erosnandi@gmail.com


# =============================================

# In the following we well set the parameters of the algoritm according to the habitats: 
# These parameters are important from habitat restoration point of view.

# The minimum number of ponds (minimum chromosome length)

min_chromosome_length = 50 # 19

# The maximum number of ponds (maximum chromosome length)
max_chromosome_length = 80 # 49

a_min = 1000 # Minimum pond area in square meters
a_max = 2000 # Maximum pond area in square meters

min_distance = 100 # The minimum distance between ponds at the initialization.

disp = 1500 # Dispersal capability of the species expressed in meters. 

# The spatial extent of the area:

x0 = 560227.9547999999485910
y0 = 544842.1668000000063330


x1 = 564087.9547999999485910

y1 = 547602.1668000000063330

# x0 = 532893.67909
# y0 = 586962.23640


# x1 = 538370.12769

# y1 = 591318.25370

# =============================================
# Let's set the parameters of the genetic algorithms

population_size = 100 # The size of the initial population


P_insert = 0.1 # The threshold value for pond insertion. 
P_reloc = 0.8 # Threshold value for relocation of a pond

# In the case above, between P_insert and P_reloc we insert a new pond, while below P_reloc we make a pond a relocation. 

# Parameters for elitist selection

prop = 0.2 # Proportion of the elits. 


nr_offsprings = 100 # The number of offsprings


# stop_statement = 0.00025 # This number marks the change in fitness over 5 generations. Below this, the algorithm stops.
stop_statement = 1e-6

max_generations = 100 # The maximum generation number, ie. the second stop statement

# The outputs of this algorithm: (i) A data tables saved as .csv in which are represented the x and y coordinates of the generated ponds, their area and the calculated radius for the post-processing.
                                # (ii) A data table saved as .csv in which are represented the max fitness values, the generation numbers and the parameters of the algorithm. - Also for post-processing. 
                                # (iii) If the 'output_figure' variable is True, then the algorithm returns a scatterplot about the optimized ponds. 

output_figure = True
seed = 4 # Setting the randomization seed

alpha_1 = 0.3
alpha_2 = 0.5
alpha_3 = 0.7

omega_1 = 0.25
omega_2 = 0.25
omega_3 = 0.25
omega_4 = 0.25

path = "csik.csv"








