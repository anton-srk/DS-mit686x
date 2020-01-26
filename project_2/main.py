import sys
import numpy as np
import matplotlib.pyplot as plt
from utils import *
from linear_regression import *
from svm import *
from softmax import *
from features import *
from kernel import *

#######################################################################
# 1. Introduction
#######################################################################

# Load MNIST data:
train_x, train_y, test_x, test_y = get_MNIST_data()
# Plot the first 20 images of the training set.
#plot_images(train_x[0:20, :])

#######################################################################
# 2. Linear Regression with Closed Form Solution
#######################################################################


def run_linear_regression_on_MNIST(lambda_factor=1):
    """
    Trains linear regression, classifies test data,
    computes test error on test set

    Returns:
        Final test error
    """
    train_x, train_y, test_x, test_y = get_MNIST_data()
    train_x_bias = np.hstack([np.ones([train_x.shape[0], 1]), train_x])
    test_x_bias = np.hstack([np.ones([test_x.shape[0], 1]), test_x])
    theta = closed_form(train_x, train_y, lambda_factor)
    test_error = compute_test_error_linear(test_x, test_y, theta)
    return test_error


#######################################################################
# 3. Support Vector Machine
#######################################################################


def run_svm_one_vs_rest_on_MNIST():
    """
    Trains svm, classifies test data, computes test error on test set

    Returns:
        Test error for the binary svm
    """
    train_x, train_y, test_x, test_y = get_MNIST_data()
    train_y[train_y != 0] = 1
    test_y[test_y != 0] = 1
    pred_test_y = one_vs_rest_svm(train_x, train_y, test_x)
    test_error = compute_test_error_svm(test_y, pred_test_y)
    return test_error


#print('SVM one vs. rest test_error:', run_svm_one_vs_rest_on_MNIST())


def run_multiclass_svm_on_MNIST():
    """
    Trains svm, classifies test data, computes test error on test set

    Returns:
        Test error for the binary svm
    """
    train_x, train_y, test_x, test_y = get_MNIST_data()
    pred_test_y = multi_class_svm(train_x, train_y, test_x)
    test_error = compute_test_error_svm(test_y, pred_test_y)
    return test_error


#print('Multiclass SVM test_error:', run_multiclass_svm_on_MNIST())

#######################################################################
# 4. Multinomial (Softmax) Regression and Gradient Descent
#######################################################################


def run_softmax_on_MNIST(temp_parameter=1):
    """
    Trains softmax, classifies test data, computes test error, and plots cost function

    Runs softmax_regression on the MNIST training set and computes the test error using
    the test set. It uses the following values for parameters:
    alpha = 0.3
    lambda = 1e-4
    num_iterations = 150

    Saves the final theta to ./theta.pkl.gz

    Returns:
        Final test error
    """
    train_x, train_y, test_x, test_y = get_MNIST_data()
    theta, cost_function_history = softmax_regression(
            train_x, train_y, temp_parameter, alpha=0.3,
            lambda_factor=1.0e-4, k=10, num_iterations=150)
    plot_cost_function_over_time(cost_function_history)
    test_error = compute_test_error(test_x, test_y, theta, temp_parameter)
    write_pickle_data(theta, "./theta.pkl.gz")

    return test_error

print('softmax test_error=', run_softmax_on_MNIST(temp_parameter=1))


#######################################################################
# 6. Changing Labels
#######################################################################

def run_softmax_on_MNIST_mod3(temp_parameter=1):
    """
    Trains Softmax regression on digit (mod 3) classifications.

    See run_softmax_on_MNIST for more info.
    """
    train_x, train_y, test_x, test_y = get_MNIST_data()
    train_y, test_y = update_y(train_y, test_y)
    theta, cost_function_history = softmax_regression(
            train_x, train_y, temp_parameter, alpha=0.3,
            lambda_factor=1.0e-4, k=10, num_iterations=150)
    test_error = compute_test_error(test_x, test_y, theta, temp_parameter)

    return test_error


#print('softmax mod3 test_error=', run_softmax_on_MNIST_mod3(temp_parameter=1))


#######################################################################
# 7. Classification Using Manually Crafted Features
#######################################################################

## Dimensionality reduction via PCA ##


n_components = 18
pcs = principal_components(train_x)
train_pca = project_onto_PC(train_x, pcs, n_components)
test_pca = project_onto_PC(test_x, pcs, n_components)
# train_pca (and test_pca) is a representation of our training (and test) data
# after projecting each example onto the first 18 principal components.

temp_parameter = 1
theta, cost_function_history = softmax_regression(
        train_pca, train_y, temp_parameter, alpha=0.3,
        lambda_factor=1.0e-4, k=10, num_iterations=150)
test_error = compute_test_error(test_pca, test_y, theta, temp_parameter)

print(test_error)
plot_PC(train_x[range(100),], pcs, train_y[range(100)])

firstimage_reconstructed = reconstruct_PC(train_pca[0, ],
                                          pcs, n_components, train_x)
plot_images(firstimage_reconstructed)
plot_images(train_x[0,])

secondimage_reconstructed = reconstruct_PC(train_pca[1, ],
                                           pcs, n_components, train_x)
plot_images(secondimage_reconstructed)
plot_images(train_x[1,])


## Cubic Kernel ##

train_pca10 = project_onto_PC(train_x, pcs, 10)
test_pca10 = project_onto_PC(test_x, pcs, 10)

train_cube = cubic_features(train_pca10)
test_cube = cubic_features(test_pca10)
# train_cube (and test_cube) is a representation of our training (and test) data
# after applying the cubic kernel feature mapping to the 10-dimensional PCA representations.

theta, cost_function_history = softmax_regression(
        train_cube, train_y, temp_parameter, alpha=0.3,
        lambda_factor=1.0e-4, k=10, num_iterations=150)
test_error = compute_test_error(test_cube, test_y, theta, temp_parameter)
