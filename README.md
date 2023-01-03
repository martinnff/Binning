# Binning
Binning method to perform data cloud classificatión on low resoruces machines.

This repository aims to provide tools for the extraction of geometric features in large point clouds and to allow the easy construction of robust classification models for this type of data.

This toolkit consists of two parts. The cloud2grid function builds a three-dimensional grid over the point cloud and distributes the weight of the points on each vertex of the grid. It also allows to distribute other variables or labels on these vertices in a weighted way.

On this grid of weights acts the function eigenFeatures. This function goes through all the vertices and for each one of them extracts a neighbourhood environment with the weights of the neighbouring points, on this matrix of weights performs the decomposition in eigenvalues and eigenvectors and extracts the eigenfeatures and the normals. When extracting these features in environments defined by a grid, the process is very fast, since it avoids calculating the neighbourhoods in each point.

Finally we can use the eigenfeatures of each node as well as the weighted variables or labels to perform prediction or classification tasks. The eigenfeatures extracted are the folowing:


  - ${\bf Linearity} = \frac{ev_1-ev_2}{ev_1} $
  - ${\bf Planarity} =\frac{ev_2-ev_3}{ev_1}$
  - ${\bf Scattering} =\frac{ev_3}{ev_1}$
  - ${\bf Surface Variation} =\frac{ev_1}{ev_1+ev_2+ev_3}$
  - ${\bf Omnivariance} =\sqrt{ev_1 \cdot ev_2 \cdot ev_3}$
  - ${\bf Anisotropy} =\frac{ev_1-ev_3}{ev_2}$
  - ${\bf Sum} =ev_1+ev_2+ev_3$


The functions were written in c++ using the Rcpp package and make use of openMP parallelisation. It also consists of wraper functions written in R to interface to this c++ code.

The images below shows the raw pointcloud of three geometric shapes and the surface variation extracted using a neighborhood cube of size 3, the surface variation is ploted on the grid nodes. This feature seems to be usefull to identify the shape of corners.

![Alt text](https://github.com/martinnff/Binning/blob/master/image2.png "Raw pointcloud")

![Alt text](https://github.com/martinnff/Binning/blob/master/image1.png "Surface variation")
