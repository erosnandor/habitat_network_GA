import numpy as np
import random
import matplotlib.pyplot as plt
import scipy as sp 
import pandas as pd
from scipy.spatial import distance_matrix
import math

import global_vars as var

min_chromosome_length = var.min_chromosome_length

max_chromosome_length = var.max_chromosome_length

a_min = var.a_min
a_max = var.a_max
min_distance = var.min_distance
disp = var.disp
x0 = var.x0
y0 = var.y0


x1 = var.x1

y1 = var.y1

population_size = var.population_size
P_insert = var.P_insert
P_reloc = var.P_reloc
prop = var.prop
nr_offsprings = var.nr_offsprings

stop_statement = var.stop_statement
max_generations = var.max_generations

seed = var.seed

alpha_1 = var.alpha_1
alpha_2 = var.alpha_2
alpha_3 = var.alpha_3

omega_1 = var.omega_1
omega_2 = var.omega_2
omega_3 = var.omega_3
omega_4 = var.omega_4

output_figure = var.output_figure

# This function returns the initial population based on a minimum distance value between the ponds. 
# We need a get_coords function which is called in the function below. 

def create_initial_population ():
    global population_size, min_chromosome_length, max_chromosome_length, x0,y0,x1,y1, a_min, a_max, min_distance
    
    population = [None]*population_size

    for i in range (population_size):
        length = random.sample(range(min_chromosome_length,max_chromosome_length), 1)
        length = int (length[0])
        ind = get_coords(x0,x1,y0,y1,min_distance,a_min,a_max,length)

        population[i] = ind
    return (population)

# This function calculates the useful distance between ponds, ie. useful_distance = distance_between_centroids - radius_pond_1 - radius_pond_2

def get_coords (x0,x1, y0, y1, min_dist,a_min, a_max, max_ponds):
    
    size = 20000
    areas = np.empty(size)
    for j in range (size):
        areas[j] = a_min + (random.random()*(a_max-a_min))
 
    
    radius = np.sqrt (areas/np.pi)

    x = np.random.uniform (low = x0, high = x1, size = size)
    y = np.random.uniform (low = y0, high = y1, size = size)

    minAllowableDistance = min_dist
    numberOfPoints = max_ponds

    # Initialize first point.
    keeperX = []
    keeperY = []
    keeperRadius = []
    keeperArea = []

    keeperX.append(x[0])
    keeperY.append(y[0])
    keeperRadius.append(radius[0])
    keeperArea.append(areas[0])
    # Try dropping down more points.

    for k in range (2, numberOfPoints):
        # Get a trial point.
        thisX = x[k]
        thisY = y[k]

        thisRadius = radius[k]
        thisArea = areas[k]
        #  See how far is is away from existing keeper points.
        x_dist = pow((keeperX-thisX),2)
        y_dist = pow((keeperY-thisY),2)
        
        distances = np.sqrt(x_dist + y_dist) - keeperRadius - thisRadius

        minDistance = np.amin(distances)
  
        if (minDistance >= minAllowableDistance):

            keeperX.append(thisX) 
            keeperY.append(thisY)
            keeperRadius.append(thisRadius)
            keeperArea.append(thisArea)

    return np.vstack ((keeperX, keeperY, keeperArea))

# I wrote here the pond insertion function because it is very similar to our initialization 'get_cords' function

def insert_pond (individ, new_ponds):
    
    global x0,x1, y0, y1, min_distance, a_min, a_max

    size = 20000
    areas = np.empty(size)
    for j in range (size):
        areas[j] = a_min + (random.random()*(a_max-a_min))
 
    
    radius = np.sqrt (areas/np.pi)

    x = np.random.uniform (low = x0, high = x1, size = size)
    y = np.random.uniform (low = y0, high = y1, size = size)

    minAllowableDistance = min_distance
    
    keeperX = individ[0]
    keeperY = individ[1]
    keeperArea = individ [2]
    keeperRadius = np.sqrt (individ[2]/np.pi) 

    exist_points = len(keeperX)

    for k in range (exist_points, exist_points+new_ponds+1):

        thisX = x[k]
        thisY = y[k]

        thisRadius = radius[k]
        thisArea = areas[k]
        #  See how far is is away from existing keeper points.
        x_dist = pow((keeperX-thisX),2)
        y_dist = pow((keeperY-thisY),2)
        
        distances = np.sqrt (x_dist + y_dist) - keeperRadius - thisRadius

        minDistance = np.amin(distances)
  
        if (minDistance >= minAllowableDistance):

            keeperX = np.append(keeperX,thisX)
            keeperY = np.append(keeperY, thisY)
            keeperRadius = np.append(keeperRadius, thisRadius)
            keeperArea = np.append(keeperArea,thisArea)

    return np.vstack ((keeperX[exist_points:], keeperY[exist_points:], keeperArea[exist_points:]))

# A detailed description the fitness function used by us can be found in our article. 

def fitness (population): 

    global disp, a_max, max_chromosome_length, omega_1, omega_2, omega_3, alpha_1, alpha_2, alpha_3

    
    pop_size = len(population)
    fit_scores = []

    for i in range(pop_size):
        x_coord = population[i][0]
        y_coord = population[i][1]
        pond_area = population[i][2]
        
        coord = np.column_stack((x_coord, y_coord))
        df = pd.DataFrame(coord, columns=['xcord', 'ycord'], index = range (len(x_coord)))

        index = range(len(pond_area))
        area_df = np.column_stack((index,pond_area))
        area_df = pd.DataFrame (area_df,columns=['Index','Area'], index = index)
        
        dist=np.array(distance_matrix(df.values, df.values))
        dist = np.exp ((-1/disp)*dist)
        dist = pd.DataFrame(dist, index=df.index, columns=df.index)

        dist = dist.rename_axis('Source').reset_index()
        dist = pd.melt(dist, id_vars='Source', value_name='Weight', var_name='Target').query('Source != Target').reset_index(drop=True)

        penalty = (dist.Weight >= 0.93).sum()
        penalty = 0.01*penalty


        Source_area = dist['Source'].map(area_df.set_index('Index')['Area'])
        Target_area = dist['Target'].map(area_df.set_index('Index')['Area'])
        
        dat = np.column_stack ((dist.Source, Source_area, Target_area, dist.Weight))
        dat = pd.DataFrame (dat, columns = ['Source','Source_area','Target_area', 'Weight'], index = range (len(Target_area)))
        
        ## Kifejezem %-ban az area-kat, így a végső súly 0-1 közötti szám lesz. 
        dat.Source_area = dat.Source_area/a_max
        dat.Target_area = dat.Target_area/a_max

        Individual_weights = dat.assign(new_col=dat.eval('Source_area * Target_area * Weight')).groupby('Source')['new_col'].agg('sum')
        # S = sum(Individual_weights)/(max_chromosome_length*(max_chromosome_length-1))
        
        pond_number = len(pond_area)
        inflection = max_chromosome_length/2


        S = sum(Individual_weights)/(pond_number*(pond_number-1))
        
        cv_area = np.std(pond_area)/np.mean(pond_area)
        p_ij_cv = np.std (dat.Weight)/np.mean(dat.Weight)

        fit = omega_1*(1-abs(alpha_1-S)) + omega_2*(1/(1 + np.exp(-0.2*(pond_number-inflection)))) + omega_3*(1-abs(alpha_2-p_ij_cv)) + omega_4*(1-(abs(alpha_3-cv_area)))
        fit = fit-penalty
        fit_scores.append(fit)

    return (fit_scores)


def mutation (pop):
    global x0,y0,x1,y1, a_min, a_max, P_reloc, P_insert

    pop_size = len(pop)
    mut_probs = [None]*pop_size

    for i in range(pop_size):    
        mut_probs[i] = random.random()


    for i in range (pop_size):

        if (mut_probs[i] <= P_insert ): #and mut_probs[i] < P_reloc

            new_member = insert_pond (pop[i], 1)

            pop[i] = np.hstack((pop[i], new_member))

        elif (mut_probs[i] >= P_reloc):

            chr_length = len(pop[i][1])

            j = random.sample(range(0,chr_length), 1)

            pop[i][0][j] = x0 + (random.random()*(x1-x0)) 
            pop[i][1][j] = y0 + (random.random()*(y1-y0))
            pop[i][2][j] = a_min + (random.random()*(a_max-a_min))

    return (pop)

def crossover (parents, nr_offsprings):
    
    offsprings = [None]*nr_offsprings

    for i in range (0, nr_offsprings, 2):
        parents_id = random.sample(range(0,len(parents)), 2)

        # The lengths of both the parent chromosomes are checked and the chromosome whose length is smaller
        # is taken as Parent 1. If both of the chromosome lengths has the same length, any one chromosome is taken as Parent 1.

        parent_1 = parents[parents_id[0]]
        parent_2 = parents[parents_id[1]]
        
        length_1 = len(parent_1[0,:]) 
        length_2 = len(parent_1[0,:])

        if (length_1 > length_2):
            tmp_parent = parent_1
            parent_1 = parent_2
            parent_2 = tmp_parent
        
        # print ("Parent 1:", len(parent_1[0,:]))
        # print ("Parent 2:", len(parent_2[0,:]))
        # Let's create the one random point based on the chromosome length of Parent 1 (due to shorter ch. length).
        
        cx_point_1 = random.randint (1, len(parent_1[0,:])-1) # I put length -1 to avoid the selection of the last gene.
        cx_point_2 = random.randint (1, len(parent_1[0,:])-1)

        if (cx_point_1 > cx_point_2):
            tmp_point = cx_point_1
            cx_point_1 = cx_point_2
            cx_point_2 = tmp_point


        part_1 = parent_1 [:, 0:cx_point_1]
        part_2 = parent_2 [:, cx_point_1:cx_point_2]
        part_3 = parent_1 [:, cx_point_2:]
    
        offsprings[i] = np.hstack ((part_1, part_2, part_3))
        
        # print ("Offspring 1: ",len(offsprings[i][0]))
        
        part_1 = parent_2 [:, 0:cx_point_1]
        part_2 = parent_1 [:, cx_point_1:cx_point_2]
        part_3 = parent_2 [:, cx_point_2:]

        offsprings[i+1] = np.hstack((part_1, part_2, part_3))
        # print ("Offspring 2: ",len(offsprings[i+1][0]))
    return (offsprings)

def elitist_selection(pop, k): # in this case the k is a top k% of the fittest individs. 

    fit_scores = fitness (pop)
    normalized_fit_scores = fit_scores/np.sum(fit_scores)
    
    stack = pd.DataFrame (normalized_fit_scores, columns = ["Normalized_Fitness"], index = range(len(fit_scores)))

    k = int(len(fit_scores)*k)
    selected_indivs = [None]*k

    elits = stack.sort_values(by = "Normalized_Fitness", ascending = False).head(k).index
    for i in range (k):

        # max_index = fit_scores.index(fit_scores[::-1][i]) ## ide be kell tenni azt, hogy sorba rendezze. 
        # max_index = fit_scores.index(max(fit_scores))
        id = elits[i]
        selected_indivs[i] = pop[id]

    return (selected_indivs)

def genetic_algorithm ():
    print ("The algorithm is running ...")

    global seed, max_generations, prop, nr_offsprings, stop_statement, a_min, a_max
    
    fitnesses = []
    j = []
    j.append(0)
    
    
    # Model parameters:

    elit_prop = []
    insert_stat = []
    reloc_stat = []

    # -----------------

    nr_offsprings_stat = []
    a_min_stat = []
    a_max_stat = []

    insert_stat.append(var.P_insert)
    elit_prop.append(var.prop)
    reloc_stat.append(var.P_reloc)

    nr_offsprings_stat.append(nr_offsprings)
    a_min_stat.append(a_min)
    a_max_stat.append(a_max)

    random.seed (seed) 

    pop = create_initial_population ()
    fit = fitness (pop)
    fitnesses.append (max(fit))

    max_index = fit.index(max(fit))

    candidate_x = pop[max_index][0]
    candidate_y = pop[max_index][1]
    candidate_area = pop[max_index][2]  

    for generation in range(max_generations):

        parents = elitist_selection(pop, prop)

        offsprings = crossover (parents=parents, nr_offsprings = nr_offsprings)
        offsprings = mutation (offsprings)

        pop = offsprings
        fit = fitness (pop)
        fitnesses.append(max(fit))

        n = generation+1
        j.append(n)

        insert_stat.append(var.P_insert)
        elit_prop.append(var.prop)
        reloc_stat.append(var.P_reloc)
        
        nr_offsprings_stat.append(nr_offsprings)
        a_min_stat.append(a_min)
        a_max_stat.append(a_max)

        if (max(fit) > fitnesses[0]):

            max_index = fit.index(max(fit))
            
            candidate_x = pop[max_index][0]
            candidate_y = pop[max_index][1]
            candidate_area = pop[max_index][2] 

        if (n > 5):
            diff = fitnesses[n]-fitnesses[n-5]
            if (diff <= stop_statement):
                print ("Stopped at generation: ", n)
                break
    
    print (" ")
    print ("============================================")
    print ('Initial fitness value:',fitnesses[0])

    print ("============================================")
    print ('Optimal fitness value:',np.sort(fitnesses)[::-1][0])

    candidate_radius = np.sqrt(candidate_area/np.pi)
    output = np.column_stack((candidate_x, candidate_y, candidate_area, candidate_radius))
    df = pd.DataFrame(output, columns=['x', 'y', 'area', 'radius'], index = range (len(candidate_x)))

    save_path = "".join (("output.csv"))
    df.to_csv (save_path)

    if (var.output_figure == True):
        plt.scatter (candidate_x,candidate_y, s=0.01 * candidate_area)
        save_path = "".join (("output.png"))
        plt.savefig(save_path)
        plt.close()

        plt.plot (j, fitnesses, linewidth = 1.5)
        plt.ylabel ("Fitness value")
        plt.xlabel ("Generation number")
        plt.show()

    result = np.column_stack((j, fitnesses, insert_stat, reloc_stat, elit_prop, nr_offsprings_stat, a_min_stat, a_max_stat))
    result = pd.DataFrame(result, columns=['Generation_number', 'Fitness', 'P_insert','P_reloc','Elit_prop', 'Nr_offsprings','Min_area', 'Max_area'], index = range (len(j)))
    return result