import pandas as pd
import numpy as np
import random
from scipy.spatial import distance_matrix
import matplotlib.pyplot as plt
import networkx as nx
import operator 


def build_habitat_network (coords, disp, plot = False):
    
    dist = distance_matrix(coords.values, coords.values)
    # dist = np.where (dist>disp, 0, np.exp((-1/disp) * dist))
    dist = np.exp ((-1/disp)*dist)
    
    max_disp = np.exp(-1)

    dist = np.where (dist < max_disp, 0, dist)

    dist = pd.DataFrame(dist, index=coords.index, columns=coords.index)
    dist = dist.rename_axis('Source').reset_index()
    dist = pd.melt(dist, id_vars='Source', value_name='Weight', var_name='Target').query('Source != Target').reset_index(drop=True)
    
    G = nx.Graph ()
    
    for x in range (len(dist.Source)):
        if (dist.Weight[x] != 0):
            G.add_edge(dist.Source[x], dist.Target[x], weight = dist.Weight[x]*3)
    
    if plot == True:
        pos=nx.spring_layout(G)
        nx.draw_networkx(G, pos, width=dist.Weight*1)
        plt.show ()
    
    return (G)


def habitat_build_from_distance(dist, disp = 1500, plot = False):

    dist = np.exp ((-1/disp)*dist)
    
    max_disp = np.exp(-1)

    dist = np.where (dist < max_disp, 0, dist)
    # print(dist)
    dist = pd.DataFrame(dist)
    
    dist = dist.rename_axis('Source').reset_index()
    dist = pd.melt(dist, id_vars='Source', value_name='Weight', var_name='Target').query('Source != Target').reset_index(drop=True)
    print(max(dist.Weight))
    G = nx.Graph ()
    
    for x in range (len(dist.Source)):
        if (dist.Weight[x] != 0):
            G.add_edge(dist.Source[x], dist.Target[x], weight = dist.Weight[x]*3)
    
    if plot == True:
        pos=nx.spring_layout(G)
        nx.draw_networkx(G, pos, width=dist.Weight*1)
        plt.show ()
    
    return (G)


def rand(g):
    """
    Performs robustness analysis based on random attack, on the network 
    specified by infile. Returns a list with fraction of nodes removed, a 
    list with the corresponding sizes of the largest component of the 
    network, and the overall vulnerability of the network.
    """

    l = [(node, 0) for node in g.nodes()]
    random.shuffle(l)
    x = []
    y = []
    largest_component = max(nx.connected_components(g), key = len)
    n = len(g.nodes())
    x.append(0)
    y.append(len(largest_component) * 1. / n)
    R = 0.0
    for i in range(1, n):
        g.remove_node(l.pop(0)[0])
        largest_component = max(nx.connected_components(g), key = len)
        x.append(i * 1. / n)
        R += len(largest_component) * 1. / n
        y.append(len(largest_component) * 1. / n)
    return x, y, 0.5 - R / n

def degree(g, recalculate = False):
    """
    Performs robustness analysis based on degree centrality,  
    on the network specified by infile using sequential (recalculate = True) 
    or simultaneous (recalculate = False) approach. Returns a list 
    with fraction of nodes removed, a list with the corresponding sizes of 
    the largest component of the network, and the overall vulnerability 
    of the network.
    """
    m = nx.degree_centrality(g)
    l = sorted(m.items(), key = operator.itemgetter(1), reverse = True)
    x = []
    y = []
    largest_component = max(nx.connected_components(g), key = len)
    n = len(g.nodes())
    x.append(0)
    y.append(len(largest_component) * 1. / n)
    R = 0.0
    for i in range(1, n - 1):
        g.remove_node(l.pop(0)[0])
        if recalculate:
            m = nx.degree_centrality(g)
            l = sorted(m.items(), key = operator.itemgetter(1), 
                       reverse = True)
        largest_component = max(nx.connected_components(g), key = len)
        x.append(i * 1. / n)
        R += len(largest_component) * 1. / n
        y.append(len(largest_component) * 1. / n)
    return x, y, 0.5 - R / n

def betweenness(g, recalculate = False):
    
    """
    Performs robustness analysis based on betweenness centrality,  
    on the network specified by infile using sequential (recalculate = True) 
    or simultaneous (recalculate = False) approach. Returns a list 
    with fraction of nodes removed, a list with the corresponding sizes of 
    the largest component of the network, and the overall vulnerability 
    of the network.
    """

    m = nx.betweenness_centrality(g)
    l = sorted(m.items(), key = operator.itemgetter(1), reverse = True)
    x = []
    y = []
    largest_component = max(nx.connected_components(g), key = len)
    n = len(g.nodes())
    x.append(0)
    y.append(len(largest_component) * 1. / n)
    R = 0.0
    for i in range(1, n):
        g.remove_node(l.pop(0)[0])
        if recalculate:
            m = nx.betweenness_centrality(g)
            l = sorted(m.items(), key = operator.itemgetter(1), 
                       reverse = True)
        largest_component = max(nx.connected_components(g), key = len)
        x.append(i * 1. / n)
        R += len(largest_component) * 1. / n
        y.append(len(largest_component) * 1. / n)
    return x, y, 0.5 - R / n

def closeness(g, recalculate = False):
    """
    Performs robustness analysis based on closeness centrality,  
    on the network specified by infile using sequential (recalculate = True) 
    or simultaneous (recalculate = False) approach. Returns a list 
    with fraction of nodes removed, a list with the corresponding sizes of 
    the largest component of the network, and the overall vulnerability 
    of the network.
    """
    m = nx.closeness_centrality(g)
    l = sorted(m.items(), key = operator.itemgetter(1), reverse = True)
    x = []
    y = []
    largest_component = max(nx.connected_components(g), key = len)
    n = len(g.nodes())
    x.append(0)
    y.append(len(largest_component) * 1. / n)
    R = 0.0
    for i in range(1, n):
        g.remove_node(l.pop(0)[0])
        if recalculate:
            m = nx.closeness_centrality(g)
            l = sorted(m.items(), key = operator.itemgetter(1), 
                       reverse = True)
        largest_component = max(nx.connected_components(g), key = len)
        x.append(i * 1. / n)
        R += len(largest_component) * 1. / n
        y.append(len(largest_component) * 1. / n)
    return x, y, 0.5 - R / n


def get_awc (df, disp = 1500, a_max = 2000):

    coords = df[["x", "y"]]

    node_nr = len(coords.x)

    dist = distance_matrix(coords.values, coords.values)
    # dist = np.where (dist>disp, 0, np.exp((-1/disp) * dist))
    dist = np.exp ((-1/disp)*dist)
    
    # max_disp = np.exp(-1)

    # dist = np.where (dist < max_disp, 0, dist)

    dist = pd.DataFrame(dist, index=coords.index, columns=coords.index)
    dist = dist.rename_axis('Source').reset_index()
    dist = pd.melt(dist, id_vars='Source', value_name='Weight', var_name='Target').query('Source != Target').reset_index(drop=True)
    
    pond_area = df.area

    index = range(len(pond_area))
    area_df = np.column_stack((index,pond_area))
    area_df = pd.DataFrame (area_df,columns=['Index','Area'], index = index)

    Source_area = dist['Source'].map(area_df.set_index('Index')['Area'])
    Target_area = dist['Target'].map(area_df.set_index('Index')['Area'])
        
    dat = np.column_stack ((dist.Source, Source_area, Target_area, dist.Weight))
    dat = pd.DataFrame (dat, columns = ['Source','Source_area','Target_area', 'Weight'], index = range (len(Target_area)))
        
    ## Kifejezem %-ban az area-kat, így a végső súly 0-1 közötti szám lesz. 
    dat.Source_area = dat.Source_area/a_max
    dat.Target_area = dat.Target_area/a_max

    Individual_weights = dat.assign(new_col=dat.eval('Source_area * Target_area * Weight')).groupby('Source')['new_col'].agg('sum')
    
    ## Ezt ki kell cserélni valahogy ezt a cuccot. ... 
    S = sum(Individual_weights)/ (node_nr*(node_nr-1))
    return S
