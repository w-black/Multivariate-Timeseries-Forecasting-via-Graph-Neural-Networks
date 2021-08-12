# Multivariate Time-series Forecasting via Graph Neural Netwokrs
### Generates low dimensional embeddings using graphSAGE for highly multivariate time-series as a method for forecasting values via KNN

Pipeline:
1. Import multivariate time-series CSV file and features for each time-series CSV file
2. Generate a time-series dataframe for each day and the previous 60 days with corresponding feature dataframe. Also generate dataframe of next 10 days
3. For each of these sub dataframes: 
    1. Calculate adjacency matrix between timeseries using their correlation to convert to network structure (each node representing a different timeseries)
    2. Generate Input files required for GCN method graphSAGE from the network structure 
    3. Train embedding of the network structure with graphSAGE
4. Calculate dissimilarities between all the embeddings that have been generated for each day using Procrustes and place in adjacency matrix and form network structure again
5. Create files necessary for graphSAGE to generate embedding from this newest network
6. Train embedding of this network using graphSAGE with each point representing one day (and its previous 60) from the master files import at stage 1
7. Then for day T find its embedding and use KNN weighted on distance to generate 10 day forecast
