# Project 2: Digit recognition (Part 1) 

MNIST digit classification __without__ neural net usage.

__linear_regression.py__ linear regression implementation  
__svm.py__ support vector machine implementation  
__softmax.py__ multinomial regression implementation  
__features.py__ principal component analysis (PCA) dimensionality reduction  
__kernel.py__ polynomial and Gaussian RBF kernels implementation  
__main.py__ digit classification by means of the methods mentioned above  
__utils.py__ helper functions  

The get_MNIST_data function from __utils.py__ returns the data in the following format:

_train_x_ : A matrix of the training data. Each row of train_x contains the features of one image, which are simply the raw pixel values flattened out into a vector of length  784=28<sup>2</sup> . The pixel values are float values between 0 and 1 (0 stands for black, 1 for white, and various shades of gray in-between).

_train_y_ : The labels for each training datapoint, aka the digit shown in the corresponding image (a number between 0-9).

_test_x_ : A matrix of the test data, formatted like train_x.

_test_y_ : The labels for the test data, which should only be used to evaluate the accuracy of different classifiers in your report.
