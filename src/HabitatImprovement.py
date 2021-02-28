import numpy as np
import random
import matplotlib.pyplot as plt
import scipy as sp 
import pandas as pd
from scipy.spatial import distance_matrix

from datetime import datetime

def insert_pond (individ, new_ponds, x0,x1, y0, y1, min_distance, a_min, a_max):

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

def create_initial_population (fixed_ponds, population_size, min_chromosome_length, max_chromosome_length, x0,y0,x1,y1, a_min, a_max, min_distance):
    
    population = [None]*population_size

    for i in range (population_size):
        length = random.sample(range(min_chromosome_length,max_chromosome_length), 1)
        length = int (length[0])
        population[i] = insert_pond (fixed_ponds, length, x0,x1, y0, y1, min_distance, a_min, a_max)

    return (population)

def fitness (population, fixed_ponds, disp, a_max, max_chromosome_length, a, alpha_1, alpha_2, alpha_3, omega_1, omega_2, omega_3, omega_4): 

    pop = population

    pop_size = len(pop)
    fit_scores = []

    ## Penalty for the fixed ponds:
    x = fixed_ponds[:,0]
    y = fixed_ponds[:,1]
    coord = np.column_stack((x, y))
    df = pd.DataFrame(coord, columns=['xcord', 'ycord'], index = range (len(x)))
    
    dist=np.array(distance_matrix(df.values, df.values))
    dist = np.exp ((-1/disp)*dist)
    dist = pd.DataFrame(dist, index=df.index, columns=df.index)

    dist = dist.rename_axis('Source').reset_index()
    dist = pd.melt(dist, id_vars='Source', value_name='Weight', var_name='Target').query('Source != Target')
    
    # fixed_areas = fixed_ponds.area

    penalty_fixed = (dist.Weight >= 0.93).sum()

    theoretical_maximum = len(x)+max_chromosome_length

    for i in range(pop_size):

        x_coord = np.append(fixed_ponds[:,0], population[i][0])
        y_coord = np.append(fixed_ponds[:,1], population[i][1])
        pond_area = np.append(fixed_ponds[:,2], population[i][2])
        
        coord = np.column_stack((x_coord, y_coord))
        df = pd.DataFrame(coord, columns=['xcord', 'ycord'], index = range (len(x_coord)))

        index = range(len(pond_area))
        area_df = np.column_stack((index,pond_area))
        area_df = pd.DataFrame (area_df,columns=['Index','Area'], index = index)
        
        dist=np.array(distance_matrix(df.values, df.values))
        dist = np.exp ((-1/disp)*dist)
        dist = pd.DataFrame(dist, index=df.index, columns=df.index)

        dist = dist.rename_axis('Source').reset_index()
        dist = pd.melt(dist, id_vars='Source', value_name='Weight', var_name='Target').query('Source != Target')

        penalty = (dist.Weight >= 0.93).sum()
        penalty = penalty-penalty_fixed

        if (penalty <0 ):
            penalty = 0
       
        penalty = 0.001*penalty


        Source_area = dist['Source'].map(area_df.set_index('Index')['Area'])
        Target_area = dist['Target'].map(area_df.set_index('Index')['Area'])
        
        dat = np.column_stack ((dist.Source, Source_area, Target_area, dist.Weight))
        dat = pd.DataFrame (dat, columns = ['Source','Source_area','Target_area', 'Weight'], index = range (len(Target_area)))
        
        ## Kifejezem %-ban az area-kat, így a végső súly 0-1 közötti szám lesz. 
        dat.Source_area = dat.Source_area/a_max
        dat.Target_area = dat.Target_area/a_max

        Individual_weights = dat.assign(new_col=dat.eval('Source_area * Target_area * Weight')).groupby('Source')['new_col'].agg('sum')

        pond_number = len(pond_area)

        S = sum(Individual_weights)/(pond_number*(pond_number-1))
        
        inflection = theoretical_maximum/2
        # inflection = max_chromosome_length/2
        
        cv_area = np.std(population[i][2]) / np.mean(population[i][2])


        p_ij_cv = np.std (dat.Weight)/np.mean(dat.Weight)

        beta_1 = (1-abs(alpha_1-S))
        beta_2 = (1/(1 + np.exp(a*(pond_number-inflection))))
        beta_3 = (1-abs(alpha_2-p_ij_cv))
        beta_4 = (1-(abs(alpha_3-cv_area)))

        fit = omega_1*beta_1 + omega_2*beta_2 + omega_3*beta_3 + omega_4*beta_4
        fit = fit - penalty
        fit_scores.append(fit)
        # if (fit < 0):
        #     print(fit_scores)

    pop = []
    return (fit_scores)

def mutation (pop, x0,y0,x1,y1, a_min, a_max, P_reloc, P_insert, min_distance):

    pop_size = len(pop)
    mut_probs = [None]*pop_size
    for i in range(pop_size):    
        mut_probs[i] = random.random()


    for i in range (pop_size):

        if (mut_probs[i] <= P_insert ): #and mut_probs[i] < P_reloc

            new_member = insert_pond (pop[i], 1, x0,x1, y0, y1, min_distance, a_min, a_max)

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

        parent_1 = parents[parents_id[0]]
        parent_2 = parents[parents_id[1]]
        
        length_1 = len(parent_1[0,:]) 
        length_2 = len(parent_1[0,:])

        if (length_1 > length_2):
            tmp_parent = parent_1
            parent_1 = parent_2
            parent_2 = tmp_parent
        
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
        
        
        
        part_1 = parent_2 [:, 0:cx_point_1]
        part_2 = parent_1 [:, cx_point_1:cx_point_2]
        part_3 = parent_2 [:, cx_point_2:]

        offsprings[i+1] = np.hstack((part_1, part_2, part_3))
        
    return (offsprings)

def elitist_selection(pop, k, fit_scores):

    normalized_fit_scores = fit_scores/np.sum(fit_scores)
    
    stack = pd.DataFrame (normalized_fit_scores, columns = ["Normalized_Fitness"], index = range(len(fit_scores)))

    k = int(len(fit_scores)*k)
    selected_indivs = [None]*k

    elits = stack.sort_values(by = "Normalized_Fitness", ascending = False).head(k).index
    
    for i in range (k):

        id = elits[i]
        selected_indivs[i] = pop[id]

    return (selected_indivs)

def genetic_algorithm (fixed_ponds, x0, y0, x1, y1, min_chromosome_length, max_chromosome_length, a, a_min, a_max, disp, min_distance, population_size, P_insert,P_reloc, prop, nr_offsprings, stop_statement, max_generations, seed, alphas, omegas):
    
    start=datetime.now()

    print("The algorithm is running ...")

    """
    The empty vectors below will be filled with the model parameters.
    The role of this will be important at the concatenation and plotting different model outputs.
    """

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

    insert_stat.append(P_insert)
    elit_prop.append(prop)
    reloc_stat.append(P_reloc)

    nr_offsprings_stat.append(nr_offsprings)
    a_min_stat.append(a_min)
    a_max_stat.append(a_max)

    np.random.seed (seed) 
    random.seed(seed)

    pop = create_initial_population (fixed_ponds, population_size, min_chromosome_length, max_chromosome_length, x0,y0,x1,y1, a_min, a_max, min_distance)
    
    fit = fitness(population = pop, fixed_ponds = fixed_ponds,  disp = disp, a_max = a_max, 
                max_chromosome_length = max_chromosome_length, a = a, 
                alpha_1 = alphas[0], alpha_2 = alphas[1], alpha_3 = alphas[2],
                omega_1 = omegas[0], omega_2 = omegas[1], omega_3 = omegas[2], omega_4 = omegas[3])
    
    fitnesses.append (max(fit))

    max_index = fit.index(max(fit))

    candidate_x = pop[max_index][0]
    candidate_y = pop[max_index][1]
    candidate_area = pop[max_index][2]  

    for generation in range(max_generations):

        parents = elitist_selection(pop, prop, fit_scores = fit)

        offsprings = crossover (parents=parents, nr_offsprings = nr_offsprings)
        
        pop = offsprings
        
        offsprings = mutation (pop, x0,y0,x1,y1, a_min, a_max, P_reloc, P_insert, min_distance)

        fit = fitness(population = pop, fixed_ponds = fixed_ponds, disp = disp, a_max = a_max, 
                max_chromosome_length = max_chromosome_length, a = a, 
                alpha_1 = alphas[0], alpha_2 = alphas[1], alpha_3 = alphas[2],
                omega_1 = omegas[0], omega_2 = omegas[1], omega_3 = omegas[2], omega_4 = omegas[3])
        
        fitnesses.append(max(fit))

        n = generation+1
        j.append(n)

        insert_stat.append(P_insert)
        elit_prop.append(prop)
        reloc_stat.append(P_reloc)
        
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
    print ('Initial fitness value:',fitnesses[0])
    print (" ")
    print ('Optimal fitness value:',np.sort(fitnesses)[::-1][0])

    candidate_radius = np.sqrt(candidate_area/np.pi)
    output = np.column_stack((candidate_x, candidate_y, candidate_area, candidate_radius))
    best_indiv = pd.DataFrame(output, columns=['x', 'y', 'area', 'radius'], index = range (len(candidate_x)))

    fitness_result = np.column_stack((j, fitnesses, insert_stat, reloc_stat, elit_prop, nr_offsprings_stat, a_min_stat, a_max_stat))
    fitness_result = pd.DataFrame(fitness_result, columns=['Generation_number', 'Fitness', 'P_insert','P_reloc','Elit_prop', 'Nr_offsprings','Min_area', 'Max_area'], index = range (len(j)))
    print("Elapsed time: ", datetime.now()-start)

    return best_indiv, fitness_result

if __name__ == "__main__":
    
    # ======= Species and habitat related parameters ======= 

    # The spatial extent of the area 

    x0 = 560227.9547999999485910
    y0 = 544842.1668000000063330

    x1 = 564087.9547999999485910
    y1 = 547602.1668000000063330

    min_chromosome_length = 50
    max_chromosome_length = 80

    a_min = 1000 # Minimum pond area in square meters
    a_max = 2000 # Maximum pond area in square meters

    disp = 1500 # Dispersal capability of the species expressed in meters. 


    # ======= Genetic Algorithm related parameters =======

    min_distance = 100 # The minimum distance between ponds at the initialization.

    population_size = 100 # The size of the initial population
    P_insert = 0.1 # The threshold value for pond insertion. 
    P_reloc = 0.8 # Threshold value for relocation of a pond.
    prop = 0.2 # Proportion of the elits. 

    nr_offsprings = 100 # The number of offsprings
    stop_statement = 1e-6
    max_generations = 100 # The maximum generation number, ie. the second stop statement

    seed = 4 # Setting the randomization seed
    a = -0.2
    alphas = [0.1, 0.5, 0.7]

    omegas = [0.25,0.25,0.25,0.25]

    fixed_ponds = pd.read_csv("ciuc.csv")

    fixed_ponds = np.column_stack((fixed_ponds.x, fixed_ponds.y, fixed_ponds.area))
    print(fixed_ponds[:,0])
    indiv, result = genetic_algorithm(fixed_ponds, x0, y0, x1, y1, min_chromosome_length, max_chromosome_length, a, a_min, a_max, disp, min_distance, population_size, P_insert,P_reloc, prop, nr_offsprings, stop_statement, max_generations, seed, alphas, omegas)
    
    plt.plot(result.Generation_number, result.Fitness)
    plt.show()
