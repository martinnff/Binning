# Binning
Binning method to perform data cloud classificatión on low resoruces machines.

This repository aims to provide tools for the extraction of geometric features in large point clouds and to allow the easy construction of robust classification models for this type of data.

This toolkit consists of two parts. The cloud2grid function builds a three-dimensional grid over the point cloud and distributes the weight of the points on each vertex of the grid. It also allows to distribute other variables or labels on these vertices in a weighted way.

On this grid of weights acts the function eigenFeatures. This function goes through all the vertices and for each one of them extracts a neighbourhood environment with the weights of the neighbouring points, on this matrix of weights performs the decomposition in eigenvalues and eigenvectors and extracts the eigenfeatures in this environment. When extracting these features in environments defined by a grid, the process is very fast, since it avoids calculating the neighbourhoods in each point.

Finally we can use the eigenfeatures of each node as well as the weighted variables or labels to perform prediction or classification tasks.

The functions were written in c++ using the Rcpp package and make use of openMP parallelisation. It consists of wraper functions written in R to interface to this c++ code.
