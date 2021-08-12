# TemPr: Multivariate Time-series Forecasting via Graph Neural Networks

## Abstract 

In the current forecasting literature there exists a large body of work
devoted to multivariate techniques which leverage the dependencies between individual
time series to produce a prediction. Despite the strengths of these methods, they
fail to take into account the interconnectivity between the time series and the resulting network architecture displayed by the total collection.
We introduce a novel multivariate forecast strategy called TemPr, that uses graph
neural networks (GNNs) to uncover low-dimensional representations of the multivariate time series, and tracks changes in this temporal structure. Additionally,
unlike existing methods, TemPr incorporates supplementary feature time series  in its forecasting pipeline. We also present the univariate sibling of this
method, UniTempr, which harnesses the power of GNNs to compare sub-intervals
of a time series in order to produce a forecast. The power in taking the multivariate
network approach is demonstrated in our results where we back-test these techniques
from 2007-2019 on the S&P500, comparing their Sharpe ratios and P&L scores. TemPr achieves
a Sharpe ratio above 2.5, on par with the performance of LSTM networks, and
significantly outperforms UniTempr, further supporting our hypothesis that it is
essential to consider the temporal state of the entire multivariate timeseries when producing forecasts.


## Pipeline
### Converts multivariate timeseries to a network structure, generates a low dimensional embedding using graphSAGE graph neural networks, then compares this temporal embedding of the timesseries to historical data to produce a forecast.
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
