# Text-Clustering
OBJECTIVE
      Implement the DBSCAN clustering algorithm. Deal with text data (news records) in document-term sparse matrix format. Think        about best metrics for evaluating clustering solutions. Design a proximity function for text data. Think about the Curse of Dimensionality.
INTRODUCTION
Without using libraries, we have to implement DBSCAN clustering algorithm. The data provided is in the form of sparse matrix. The other important task is to find the optimal value for eps(threshold) for DBSCAN algorithm. Input data (provided as training data) consists of 8580 text records in sparse format. No labels are provided.
SCOPE
  • Find a way to deal with sparse data.
  • Create a CSR matrix.
  • Find a way to reduce dimensionality
  • Find a way to find effective eps value
  • Implement DBSCAN METHODOLOGY
  • SPLIT DATA AND CONVERT IT TO CSR MATRIX
      First, I splitted the data into two list valslist1 and valslist2, respectively. Then I created csr matrix using activity 2(k-NN). Next step is to normalize the matrix to a standard scale. The matrix is of dimensions 8650*125680 approx.
  • DIMENSIONALITY REDUCTION
      Next step is to reduce the dimensionality of data as the data is quite large and was facing a lot of issues while dealing with it. I have used random projection and TruncateSVD for this, as PCA does not work on Sparse data. Among, Random Projection and TruncateSVD, TruncateSVD performs better and number of components selected are random by hit and trial method.
 tsvd = TruncatedSVD(n_components=150)
 X_sparse_tsvd = tsvd.fit(mat2).transform(mat2)
  • FINDING EPS(threshold) FOR DBSCAN
      I have used nearestneighbour algorithm to find k neighbors and sorted them in descending order. The value of k is randomly selected. I have tried elbow method for finding optimal value of k but that didn’t work. The plot after sorting distances is as shown below which shows ideal value is 0.33 approx.
  • FINDING DISTANCE BETWEEN DIFFERENT POINTS
      I have used Euclidean distance to calculate distance between different points and plotted a graph which shows that data is  densely clustered. I have also used this proximity measure to find distance between any two points while creating clusters in DBSCAN

 • IMPLEMENTING DBSCAN
The most important part of this assignment is to implement DBSCAN from scratch.
Implementing DBSCAN is multi-way process. I have used the same value of MinPts as I used to find and finally settled on 7. Eps is as mentioned above is 0.33 approx. Any point that lies within the eps and have points greater than MinPts under the eps radius is considered Core point and if any of these two condition fails than it is a noise point (-1). I have tried different ways of solving DBSCAN problem but
