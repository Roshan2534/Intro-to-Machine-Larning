import os
import csv
import time
import pickle
import numpy as np
from scipy.optimize import minimize
from scipy.io import loadmat
from math import sqrt


def initializeWeights(n_in, n_out):
    """
    # initializeWeights return the random weights for Neural Network given the
    # number of node in the input layer and output layer

    # Input:
    # n_in: number of nodes of the input layer
    # n_out: number of nodes of the output layer
       
    # Output: 
    # W: matrix of random initial weights with size (n_out x (n_in + 1))"""

    epsilon = sqrt(6) / sqrt(n_in + n_out + 1)
    W = (np.random.rand(n_out, n_in + 1) * 2 * epsilon) - epsilon
    return W


def sigmoid(z):
    """# Notice that z can be a scalar, a vector or a matrix
    # return the sigmoid of input z"""

    return 1.0/(np.exp(-z)+1.0) # your code here


def preprocess():
    """ Input:
     Although this function doesn't have any input, you are required to load
     the MNIST data set from file 'mnist_all.mat'.

     Output:
     train_data: matrix of training set. Each row of train_data contains 
       feature vector of a image
     train_label: vector of label corresponding to each image in the training
       set
     validation_data: matrix of training set. Each row of validation_data 
       contains feature vector of a image
     validation_label: vector of label corresponding to each image in the 
       training set
     test_data: matrix of training set. Each row of test_data contains 
       feature vector of a image
     test_label: vector of label corresponding to each image in the testing
       set

     Some suggestions for preprocessing step:
     - feature selection"""

    mat = loadmat('mnist_all.mat')  # loads the MAT object as a Dictionary

    # Split the training sets into two sets of 50000 randomly sampled training examples and 10000 validation examples. 
    # Your code here.
    training_set = []
    training_label = []

    test_data = []
    test_label = []

    validation_data = []
    validation_label = []

    for i in range(10):
        # ---------- training_set ------------ #
        # this will get particular digit
        _train = "train"+str(i)
        # gets full data
        train_data = mat[_train]
        # gives out shape of whole data
        train_shape = train_data.shape[0]

        # we will add those sets in our data
        training_set.extend(train_data)
        training_label.extend([i]*train_shape)

        # ---------- testing_set ------------ #
        _test = "test"+str(i)
        test_ = mat[_test]
        test_shape = test_.shape[0]

        test_data.extend(test_)
        test_label.extend([i]*test_shape)

    # converting to array
    training_set = np.asarray(training_set)
    training_label = np.asarray(training_label)

    test_data = np.asarray(test_data)
    test_label = np.asarray(test_label)
    # np.arange(60000)
    train_rand = np.arange(len(training_label))
    np.random.shuffle(train_rand)

    # shuffle train index
    n_train = train_rand[:50000]
    # basically creates validation set
    n_val = np.setdiff1d(train_rand, n_train)

    train_data, validation_data = training_set[n_train,:], training_set[n_val,:]
    train_label, validation_label = training_label[n_train], training_label[n_val]
    

    # Feature selection
    # Your code here.
    
    global feature_selected
    
    # combined the whole dataset
    combined_data = np.vstack((train_data, test_data, validation_data))
    
    # convert  to array
    full_data = np.array(combined_data)
    
    # select features
    sf = np.all(full_data == full_data[0, :], axis=0) # finds out columns that needed to be removed
    # to  be dumped into pickel file 
    feature_selected = np.where(sf == False)[0]
    
    # print(feature_selected)
    remove_selected = np.where(sf == True)

    train_data = np.delete(train_data, remove_selected, axis=1)
    validation_data = np.delete(validation_data, remove_selected, axis=1)
    test_data = np.delete(test_data, remove_selected, axis=1)
    
    train_data = train_data / 255.0
    validation_data = validation_data / 255.0
    test_data = test_data / 255.0
    
    print('preprocess done')

    return train_data, train_label, validation_data, validation_label, test_data, test_label


def nnObjFunction(params, *args):
    """% nnObjFunction computes the value of objective function (negative log 
    %   likelihood error function with regularization) given the parameters 
    %   of Neural Networks, thetraining data, their corresponding training 
    %   labels and lambda - regularization hyper-parameter.

    % Input:
    % params: vector of weights of 2 matrices w1 (weights of connections from
    %     input layer to hidden layer) and w2 (weights of connections from
    %     hidden layer to output layer) where all of the weights are contained
    %     in a single vector.
    % n_input: number of node in input layer (not include the bias node)
    % n_hidden: number of node in hidden layer (not include the bias node)
    % n_class: number of node in output layer (number of classes in
    %     classification problem
    % training_data: matrix of training data. Each row of this matrix
    %     represents the feature vector of a particular image
    % training_label: the vector of truth label of training images. Each entry
    %     in the vector represents the truth label of its corresponding image.
    % lambda: regularization hyper-parameter. This value is used for fixing the
    %     overfitting problem.
       
    % Output: 
    % obj_val: a scalar value representing value of error function
    % obj_grad: a SINGLE vector of gradient value of error function
    % NOTE: how to compute obj_grad
    % Use backpropagation algorithm to compute the gradient of error function
    % for each weights in weight matrices.

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % reshape 'params' vector into 2 matrices of weight w1 and w2
    % w1: matrix of weights of connections from input layer to hidden layers.
    %     w1(i, j) represents the weight of connection from unit j in input 
    %     layer to unit i in hidden layer.
    % w2: matrix of weights of connections from hidden layer to output layers.
    %     w2(i, j) represents the weight of connection from unit j in hidden 
    %     layer to unit i in output layer."""

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


def nnPredict(w1, w2, data):
    """% nnPredict predicts the label of data given the parameter w1, w2 of Neural
    % Network.

    % Input:
    % w1: matrix of weights of connections from input layer to hidden layers.
    %     w1(i, j) represents the weight of connection from unit j in input 
    %     layer to unit i in hidden layer.
    % w2: matrix of weights of connections from hidden layer to output layers.
    %     w2(i, j) represents the weight of connection from unit j in hidden 
    %     layer to unit i in output layer.
    % data: matrix of data. Each row of this matrix represents the feature 
    %       vector of a particular image
       
    % Output: 
    % label: a column vector of predicted labels"""

    # labels = np.array([])
    # Your code here
    
    # get dataset size
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


"""**************Neural Network Script Starts here********************************"""

train_data, train_label, validation_data, validation_label, test_data, test_label = preprocess()
now_time = time.time()
start_time = time.strftime("%X")
print(start_time)


#  Train Neural Network

# set the number of nodes in input unit (not including bias unit)
n_input = train_data.shape[1]

# set the number of nodes in hidden unit (not including bias unit)
n_hidden = 50

# set the number of nodes in output unit
n_class = 10

# initialize the weights into some random matrices
initial_w1 = initializeWeights(n_input, n_hidden)
initial_w2 = initializeWeights(n_hidden, n_class)

# unroll 2 weight matrices into single column vector
initialWeights = np.concatenate((initial_w1.flatten(), initial_w2.flatten()), 0)

# set the regularization hyper-parameter
lambdaval = 0

args = (n_input, n_hidden, n_class, train_data, train_label, lambdaval)

# Train Neural Network using fmin_cg or minimize from scipy,optimize module. Check documentation for a working example

opts = {'maxiter': 50}  # Preferred value.

nn_params = minimize(nnObjFunction, initialWeights, jac=True, args=args, method='CG', options=opts)

# In Case you want to use fmin_cg, you may have to split the nnObjectFunction to two functions nnObjFunctionVal
# and nnObjGradient. Check documentation for this function before you proceed.
# nn_params, cost = fmin_cg(nnObjFunctionVal, initialWeights, nnObjGradient,args = args, maxiter = 50)


# Reshape nnParams from 1D vector into w1 and w2 matrices
w1 = nn_params.x[0:n_hidden * (n_input + 1)].reshape((n_hidden, (n_input + 1)))
w2 = nn_params.x[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))

# Test the computed parameters

predicted_label = nnPredict(w1, w2, train_data)

# find the accuracy on Training Dataset

print('\n Training set Accuracy:' + str(100 * np.mean((predicted_label == train_label).astype(float))) + '%')

predicted_label = nnPredict(w1, w2, validation_data)

# find the accuracy on Validation Dataset

print('\n Validation set Accuracy:' + str(100 * np.mean((predicted_label == validation_label).astype(float))) + '%')

predicted_label = nnPredict(w1, w2, test_data)

# find the accuracy on Validation Dataset

print('\n Test set Accuracy:' + str(100 * np.mean((predicted_label == test_label).astype(float))) + '%')

end_time = time.strftime("%X")
execution_time = str(round(time.time() - now_time, 2))
print(execution_time)
# ye last mein uncomment karna submit karne ke time
# basically jitna data hai woh pura pickel file mein aajaega
# fir submit kardena pickel file

# obj = [feature_selected, n_hidden, w1, w2, lambdaval]        
# pickle.dump(obj, open('params.pickle', 'wb'))