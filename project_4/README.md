# Project 4: Collaborative Filtering via Gaussian Mixtures

Building a mixture model for collaborative filtering. A data matrix containing movie ratings made by users where the matrix is extracted from a much larger Netflix database is given. Any particular user has rated only a small fraction of the movies so the data matrix is only partially filled. The goal is to predict all the remaining entries of the matrix.

Mixtures of Gaussians are used to solve this problem. The model assumes that each user's rating profile is a sample from a mixture model. In other words, there exist _K_ possible types of users and, in the context of each user, a user type must be sampled and then the rating profile from the Gaussian distribution associated with the type. The Expectation Maximization (EM) algorithm is used to estimate such a mixture from a partially observed rating matrix. The EM algorithm proceeds by iteratively assigning (softly) users to types (E-step) and subsequently re-estimating the Gaussians associated with each type (M-step). Once the mixture is obtained, it can be used to predict values for all the missing entries in the data matrix.

The code is organized as follows:

* __kmeans.py__ contains a baseline using the K-means algorithm  
* __naive_em.py__ a first version of the EM algorithm  
* __em_log.py__ a mixture model for collaborative filtering  
* __common.py__ contains the common functions for all the models  

The data files:

* _toy_data.txt_ a 2D dataset used for naive_em implementation  
* _netflix_incomplete.txt_ the netflix dataset with missing entries to be completed  
* _netflix_complete.txt_ the netflix dataset with missing entries completed  
* _test_incomplete.txt_, _test_complete.txt_, _test_solutions.txt_ test datasets  
