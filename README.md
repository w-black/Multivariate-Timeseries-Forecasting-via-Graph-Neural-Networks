# Multivariate Time-series Embeddings
### Generates low dimensional embeddings for highly multivariate time-series as a methods for forecasting values via KNN
Pipeline:
1. Input multivariate time-series CSV file and features for each time-series CSV file
2. Generate a time-series dataframe for each day and the previous 60 days with corresponding feature dataframe
3. For each of these sub dataframes, calculate adjacency matrix and network structure between timeseries using correlation
4. Generate Input files required for GCN method graphsage from this network structure 

1.  Step 1
2.  Step 2
3.  Step 3
    1.  Step 3.1
    2.  Step 3.2
    3.  Step 3.3
