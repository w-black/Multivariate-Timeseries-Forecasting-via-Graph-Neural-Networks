# Function generates network from 60-day lookback stock time-series from a specified date
# Inputs: Multivariate time series (T x N dataframe), volume and beta feature timeseries (both T x N dataframes), date
# Outputs: Converts timeseries to a network structure and the following files: Network json file, network node to ticker id map , node class map, node features, random walks from each node. These are the input files required for GraphSAGE


import pandas as pd 
import numpy as np 
import statsmodels.api as sm
import itertools
import math
from scipy.spatial import procrustes
import networkx as nx
from networkx.readwrite import json_graph
import json
import random
import ast
import os
import csv
from pathlib import Path
import shutil


def gen_gsage_inputs(timeseries, features, directory, date):
    if date==None: date='yyyymmdd' # set fall back options 
    if directory==None: directory="/users/Billy/Documents/Part_C/Diss/Code/notebooks/LIVE_2007_2019_daily_returns_data/"
    date_prefix = str(date) #so we can save embeddings for each day with different names
    daily_ret = timeseries
    feature_df = features
    print('Compiling Input sets..')
    
    ######### CREATING NETWORK FROM TIMESERIES ##########
    
    # Creat Adjacency Matrix from correlation between timeseries
    adj_matrix = daily_ret.corr()
    adj_matrix[adj_matrix < 0.4] = 0
    adj_matrix = adj_matrix.dropna(how='all').dropna(axis=1, how='all')

    # Creating graph object G
    G = nx.from_numpy_matrix(np.matrix(adj_matrix))

    # Assigns each ticker id to consecutive integers and zips together into a dictionary
    ticker_list = list(adj_matrix.columns)
    id_list = list(G.nodes())
    ticker_id_map = dict(zip(ticker_list, id_list))
    
    ######### (1) CREATING G.JSON #########
    print('Creating -G.Json')
    
    # Creating 'test' and 'val' attributes list
    int_ticker_list = G.nodes()
    train_size = math.ceil(0.7*len(int_ticker_list)) 

    test_false_list = [False]*train_size
    test_true_list = [True]*(len(int_ticker_list)-train_size)

    test_list = test_false_list + test_true_list
    val_list = [False]*len(int_ticker_list) # for now setting all to 'false'

    # Creates list of class ids -- all set to 1 as does not affect my pipeline
    class_list = [1]*len(int_ticker_list)

    # Zipping dictionaries to node ids
    test_dict = dict(zip(int_ticker_list, test_list))
    val_dict = dict(zip(int_ticker_list, val_list))
    class_dict = dict(zip(int_ticker_list, class_list))

    # Adding attributes to the nodes in the graph
    nx.set_node_attributes(G, "test", test_dict)
    nx.set_node_attributes(G, "val", val_dict )
    nx.set_node_attributes(G, "label", class_dict)

    # Formatting to correct format for JSON file
    ticker_returns_G = json_graph.node_link_data(G)
    ticker_returns_G_graph = json_graph.node_link_graph(ticker_returns_G)
    ticker_returns_G_graph.graph = {"name": "disjoint_union( ,  )"} 
    ticker_returns_G1 = json_graph.node_link_data(ticker_returns_G_graph)
    del ticker_returns_G1['multigraph']

    # Converts to correct dictionary format for JSON load 
    ticker_returns_G = json.dumps(ast.literal_eval(str(ticker_returns_G1))) 

    # Defines new directory for store output data files
    out_dir = directory
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # Creates file in new directory –– note it writes over previous 
    f = open(directory+date_prefix+"_daily_returns-G.json","w")
    f.write(str(ticker_returns_G))
    f.close()
    
    ######### (2) CREATING ID_MAP.JSON #########
    print('Creating -id_map.Json')

    # create a second map to satisfy graphsage input - this just maps integer to integer
    id_list2 = list(range(len(id_list)))
    ticker_id_map2 = dict(zip(id_list, id_list2))

    # Formats to json type
    ticker_returns_id_map = json.dumps(ticker_id_map2)

    # Opens file and writes dictionary data to file
    f = open(directory+date_prefix+"_daily_returns-id_map.json","w")
    f.write(ticker_returns_id_map)
    f.close()
    
    ######### (3) CREATING CLASS_MAP.JSON #########
    print('Creating -class_map.Json')


    # Converts class_dict to json format
    ticker_returns_class_map = json.dumps(class_dict)

    # Opens file and writes dictionary data to file
    f = open(directory+date_prefix+"_daily_returns-class_map.json","w")
    f.write(ticker_returns_class_map)
    f.close()
    
    ######### (4) CREATING FEATS.NPY ##########
    print('Creating -feats.npy')

    
    # Selecting features to use
    static_features = np.array(feature_df)
    # Opens file and writes dictionary data to file
    np.save(directory+date_prefix+"_daily_returns-feats.npy",
            static_features)

    
    ######### (5) CREATING WALKS.TXT ##########
    print('Creating -walks.txt')
    # Generates random walks 
    WALK_LEN=5
    N_WALKS=50

    G_new = json_graph.node_link_graph(ticker_returns_G1)
    nodes = int_ticker_list
    pairs = []
    for count, node in enumerate(nodes):
        if G_new.degree(node) == 0:
            continue
        for i in range(N_WALKS):
            curr_node = node
            for j in range(WALK_LEN):
                next_node = random.choice(list(G_new.neighbors(curr_node)))
                # self co-occurrences are useless
                if curr_node != node:
                    pairs.append((node,curr_node))
                    #pairs.append((ticker_id_map.get(node), ticker_id_map.get(curr_node)))
                curr_node = next_node
    # writes to txt file
    with open(directory+date_prefix+"_daily_returns-walks.txt", "w") as output:
        output.write('\n'.join('%s\t%s' % x for x in pairs))
        
    print('Files compiled and stored in '+directory)