'''
Comparing single layer MLP with deep MLP (using TensorFlow)
'''

import os
import csv
import time
import pickle
import numpy as np
from scipy.optimize import minimize
from scipy.io import loadmat
from math import sqrt

# Do not change this
def initializeWeights(n_in,n_out):
    """
    # initializeWeights return the random weights for Neural Network given the
    # number of node in the input layer and output layer

    # Input:
    # n_in: number of nodes of the input layer
    # n_out: number of nodes of the output layer
                            
    # Output: 
    # W: matrix of random initial weights with size (n_out x (n_in + 1))"""
    epsilon = sqrt(6) / sqrt(n_in + n_out + 1);
    W = (np.random.rand(n_out, n_in + 1)*2* epsilon) - epsilon;
    return W



# Replace this with your sigmoid implementation
def sigmoid(z):
    return 1.0/(np.exp(-z)+1.0)
    
# Replace this with your nnObjFunction implementation
def nnObjFunction(params, *args):
    
    n_input, n_hidden, n_class, training_data, training_label, lambdaval = args

    w1 = params[0:n_hidden * (n_input + 1)].reshape((n_hidden, (n_input + 1)))
    w2 = params[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))
    obj_val = 0

    # Your code here
    #
    #
    #
    #
    #
    
    ''' Feed Forward starts here '''
    # INPUT --> HIDDEN
    
    # how much training data we got?
    training_data_size = training_data.shape[0]
    # print(n_training_data)
    
    # adding bias
    bias = np.random.rand(training_data_size, 1) - 0.5 # creating baises that will be appended to our training_data.
    # print(bias.shape)
    
    # makeing tempcopy of our training_data, so if we mess up we can get it back.
    temp_data = training_data
    # print(temp_data.shape)
    
    # adding baises that we created to our temp_data columnwise.
    temp_data = np.column_stack((bias, temp_data))
    
    # feedforward propogation to hidden layer
    # a_h = w.T * data
    a_hidden = np.dot(temp_data, w1.transpose())
    
    # z_h = sigmoid(a_h)
    z_hidden = sigmoid(a_hidden)
    
    
    # HIDDEN --> OUTPUT
    # this will be same as previous bias
    hidden_bias = np.random.rand(z_hidden.shape[0], 1) - 0.5

    # we will again add baises to our input layer ie. z_hidden
    z_hidden = np.column_stack((hidden_bias, z_hidden))
    
    # for op layers
    a_output = np.dot(z_hidden, w2.transpose())
    
    # z_output = sigmoid(a_output)
    z_output = sigmoid(a_output)
    
    #Feed Forward completed here
    
    # ERROR FUNCTION - Negative log-likelihood error function
    y = np.zeros((training_data_size, n_class)) # 50000 x 10
    
    for i in range(training_data_size):
        y[i][training_label[i]] = 1
    
    # negllerror =         y *  ln(z_op)         +   (1-y)  *  ln(1 - z_op))
    
    y_diff = (1.0-y)
    z_op_diff = (1.0-z_output)
    
    z_op_log = np.log(z_output)
    z_op_diff_log = np.log(z_op_diff)
    
    error = np.sum(np.multiply(y, z_op_log) + np.multiply(y_diff, z_op_diff_log)) / ((-1)*training_data_size)
    # neg_ll_error = np.sum((y * np.log(z_output)) + ((1.0-y) * np.log(1.0-z_output)))
    # error = (-1.0 / training_data.shape[0])* neg_ll_error
    
    ''' BACKPROPOGATION STARTS HERE '''
    delta = z_output - y
    g_w2 = np.dot(delta.transpose(), z_hidden)
    
    temp = np.dot(delta, w2) * (z_hidden * (1.0-z_hidden))
    g_w1 = np.dot(temp.transpose(), temp_data)
    g_w1 = g_w1[1:, :]
    
    
    ''' REGULARIZATION '''
    reg = lambdaval * (np.sum(np.square(w1)) + np.sum(np.square(w2))) / (2*training_data_size)
    obj_val = error + reg
    
    g_w1_reg = np.add(g_w1, (lambdaval*w1)) / training_data_size
    g_w2_reg = np.add(g_w2, (lambdaval*w2)) / training_data_size
    
    # Make sure you reshape the gradient matrices to a 1D array. for instance if your gradient matrices are grad_w1 and grad_w2
    # you would use code similar to the one below to create a flat array
    # obj_grad = np.concatenate((grad_w1.flatten(), grad_w2.flatten()),0)
    obj_grad = np.concatenate((g_w1_reg.flatten(), g_w2_reg.flatten()),0)
    
    return (obj_val, obj_grad)

    
# Replace this with your nnPredict implementation
def nnPredict(w1,w2,data):
    
    data_size = data.shape[0]
    
    # random bias added
    bias = np.random.rand(data_size, 1)-0.5
    
    # stack 2 columns together
    temp_data = np.column_stack((bias, data))
    
    # running through hidden layer
    a_hidden = np.dot(temp_data, w1.transpose())
    
    # get sigmoid val of hidden node value
    z_hidden = sigmoid(a_hidden)
    
    hidden_bias = np.random.rand(z_hidden.shape[0], 1)-0.5
    # hidden_bias = np.full((z_hidden.shape[0], 1), 1)
    z_hidden = np.column_stack((hidden_bias, z_hidden))
    a_output = np.dot(z_hidden, w2.transpose())
    z_output = sigmoid(a_output)
    
    
    # we want the maximum value so we used argmax func.
    labels = np.argmax(z_output, axis=1)
    return labels


# Do not change this
def preprocess():
    pickle_obj = pickle.load(file=open('face_all.pickle', 'rb'))
    features = pickle_obj['Features']
    labels = pickle_obj['Labels']
    train_x = features[0:21100] / 255
    valid_x = features[21100:23765] / 255
    test_x = features[23765:] / 255

    labels = labels[0]
    train_y = labels[0:21100]
    valid_y = labels[21100:23765]
    test_y = labels[23765:]
    return train_x, train_y, valid_x, valid_y, test_x, test_y

"""**************Neural Network Script Starts here********************************"""
train_data, train_label, validation_data, validation_label, test_data, test_label = preprocess()
#  Train Neural Network
# set the number of nodes in input unit (not including bias unit)
n_input = train_data.shape[1]
# set the number of nodes in hidden unit (not including bias unit)
n_hidden = 256
# set the number of nodes in output unit
n_class = 2

# initialize the weights into some random matrices
initial_w1 = initializeWeights(n_input, n_hidden);
initial_w2 = initializeWeights(n_hidden, n_class);
# unroll 2 weight matrices into single column vector
initialWeights = np.concatenate((initial_w1.flatten(), initial_w2.flatten()),0)
# set the regularization hyper-parameter
lambdaval = 10;
args = (n_input, n_hidden, n_class, train_data, train_label, lambdaval)

#Train Neural Network using fmin_cg or minimize from scipy,optimize module. Check documentation for a working example
opts = {'maxiter' :50}    # Preferred value.

nn_params = minimize(nnObjFunction, initialWeights, jac=True, args=args,method='CG', options=opts)
params = nn_params.get('x')
#Reshape nnParams from 1D vector into w1 and w2 matrices
w1 = params[0:n_hidden * (n_input + 1)].reshape( (n_hidden, (n_input + 1)))
w2 = params[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))

#Test the computed parameters
predicted_label = nnPredict(w1,w2,train_data)
#find the accuracy on Training Dataset
print('\n Training set Accuracy:' + str(100*np.mean((predicted_label == train_label).astype(float))) + '%')
predicted_label = nnPredict(w1,w2,validation_data)
#find the accuracy on Validation Dataset
print('\n Validation set Accuracy:' + str(100*np.mean((predicted_label == validation_label).astype(float))) + '%')
predicted_label = nnPredict(w1,w2,test_data)
#find the accuracy on Validation Dataset
print('\n Test set Accuracy:' +  str(100*np.mean((predicted_label == test_label).astype(float))) + '%')
