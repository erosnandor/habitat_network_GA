import ga_functions as ga
import matplotlib.pyplot as plt
import global_vars as var
import numpy as np
import pandas as pd

# var.output_figure = False

# P_reloc =  [0.2, 0.4, 0.6, 0.8, 1]

# data = []

# for i in range(len(P_reloc)):
#     print (" ")
#     print ("P reloc: ", P_reloc[i])
#     var.P_reloc = P_reloc[i]
#     data.append(ga.genetic_algorithm())

# data = np.vstack(data)
# data = pd.DataFrame(data, columns=['Generation_number', 'Fitness', 'P_insert','P_reloc','Elit_prop', 'Nr_offsprings','Min_area', 'Max_area'])

# data.to_csv ("p_reloc_proportions.csv")


# P_insert =  [0.02, 0.04, 0.06, 0.08, 0.1]

# data = []

# for i in range(len(P_insert)):
#     print (" ")
#     print ("P insert: ", P_insert[i])
#     var.P_insert = P_insert[i]
#     data.append(ga.genetic_algorithm())

# data = np.vstack(data)
# data = pd.DataFrame(data, columns=['Generation_number', 'Fitness', 'P_insert','P_reloc','Elit_prop', 'Nr_offsprings','Min_area', 'Max_area'])

# data.to_csv ("p_insert_proportions.csv")


# P_elit =  [0.2, 0.4, 0.6, 0.8, 1]

# data = []

# for i in range(len(P_elit)):
#     print (" ")
#     print ("P elit: ", P_elit[i])
#     var.prop = P_elit[i]
#     data.append(ga.genetic_algorithm())

# data = np.vstack(data)
# data = pd.DataFrame(data, columns=['Generation_number', 'Fitness', 'P_insert','P_reloc','Elit_prop', 'Nr_offsprings','Min_area', 'Max_area'])

# data.to_csv ("elit_proportions.csv")

# var.prop = 0.2
# var.P_insert = 0.02
# var.P_reloc = 0.8
# var.max_generations = 6
# var.output_figure = True


ga.genetic_algorithm()