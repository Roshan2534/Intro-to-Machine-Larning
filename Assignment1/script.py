import numpy as np
from scipy.optimize import minimize
from scipy.io import loadmat
from numpy.linalg import det, inv
from math import sqrt, pi
import scipy.io
import matplotlib.pyplot as plt
import pickle
import sys

def ldaLearn(X,y):
    # Inputs
    # X - a N x d matrix with each row corresponding to a training example
    # y - a N x 1 column vector indicating the labels for each training example
    #
    # Outputs
    # means - A d x k matrix containing learnt means for each of the k classes
    # covmat - A single d x d learnt covariance matrix 
    
    # IMPLEMENT THIS METHOD
    Xp = np.shape(X)[1]
    means = np.zeros(shape=(Xp , np.unique(y).size))
    covmat = np.zeros(shape=(Xp, Xp))
    for i in range(np.unique(y).size):
        index = np.where(y==i)[0]
        data = X[index,:]
        means[:, i-1] = np.mean(data, axis=0).transpose()
    covmat = np.cov(X.T)
    # IMPLEMENT THIS METHOD
    return means, covmat

#def qdaLearn(X,y):
    # Inputs
    # X - a N x d matrix with each row corresponding to a training example
    # y - a N x 1 column vector indicating the labels for each training example
    #
    # Outputs
    # means - A d x k matrix containing learnt means for each of the k classes
    # covmats - A list of k d x d learnt covariance matrices for each of the k classes
    
    # IMPLEMENT THIS METHOD
    #Xy = np.concatenate((y, X), axis = 1)
    #uniqueClass = int(np.max(y))
    #d = Xy.shape[1]
    #means = np.empty((d, uniqueClass))
    #covmats = np.zeros(d)
    #for i in range(1, uniqueClass+1):
        #means[:, i] = Xy[Xy[:, 0] == np.unique(y)[i], :].mean(0)
        #covmats[i] = np.cov(Xy[Xy[:, 0] == np.unique(y)[i], 1:].T)
    
    #return means,covmats

def ldaTest(means,covmat,Xtest,ytest):
    # Inputs
    # means, covmat - parameters of the LDA model
    # Xtest - a N x d matrix with each row corresponding to a test example
    # ytest - a N x 1 column vector indicating the labels for each test example
    # Outputs
    # acc - A scalar accuracy value
    # ypred - N x 1 column vector indicating the predicted labels

    # IMPLEMENT THIS METHOD
    inverseCov = np.linalg.inv(covmat)
    # means=np.transpose(means)
    classCount = np.shape(means)[1]
    correctCount = 0.0
    N = np.shape(Xtest)[0]
    ypred = np.zeros(shape=(Xtest.shape[0], 1))
    for i in range(1, N+1):
        p = 0
        numClass = 0
        # different frm Xtest this is transpose of Xtest
        xTransposeTest = np.transpose(Xtest[i-1,:])
        for j in range(1, classCount+1):
            uptoMean = means[:, j-1]
            res = np.exp((-1/2)*np.dot(np.dot(np.transpose(xTransposeTest - uptoMean), inverseCov), (xTransposeTest - uptoMean)))
            if res > p:
                numClass = j
                p = res
                ypred[i-1,:] = j
        if numClass == ytest[i-1]:
            correctCount += 1
        
    acc = correctCount / N
    return acc,ypred

#def qdaTest(means,covmats,Xtest,ytest):
    # Inputs
    # means, covmats - parameters of the QDA model
    # Xtest - a N x d matrix with each row corresponding to a test example
    # ytest - a N x 1 column vector indicating the labels for each test example
    # Outputs
    # acc - A scalar accuracy value
    # ypred - N x 1 column vector indicating the predicted labels

    # IMPLEMENT THIS METHOD
    #return acc,ypred

def learnOLERegression(X,y):
    # Inputs:                                                         
    # X = N x d 
    # y = N x 1                                                               
    # Output: 
    # w = d x 1 
	
    # IMPLEMENT THIS METHOD
    w = np.zeros(shape=(X.shape[1], 1))
    w = np.dot(inv(np.dot(X.T, X)), np.dot(X.T, y))
    # print(w)
    return w

def learnRidgeRegression(X,y,lambd):
    # Inputs:
    # X = N x d                                                               
    # y = N x 1 
    # lambd = ridge parameter (scalar)
    # Output:                                                                  
    # w = d x 1                                                                

    # IMPLEMENT THIS METHOD
    # ridge regression = w = inv[(X.T.X + lambd.I)].X.T.y
    
    # X.T.X
    xTx = np.dot(X.T, X)
    # lamdb.I
    lamI = lambd * np.identity(X.shape[1])
    # inv(sum(above var))
    w = np.linalg.inv(xTx + lamI)
    # X.T.y
    w = np.dot(np.dot(w, X.T), y)
    return w

def testOLERegression(w,Xtest,ytest):
    # Inputs:
    # w = d x 1
    # Xtest = N x d
    # ytest = X x 1
    # Output:
    # mse
    
    # IMPLEMENT THIS METHOD
    # inside -> summation of (yi - w.T * xi)**2
    inside = ((ytest - np.dot(w.T, Xtest.T).T)**2).sum()
    # mse = 1/N * inside
    mse = (1/ ytest.shape[0]) * inside
    return mse

def regressionObjVal(w, X, y, lambd):

    # compute squared error (scalar) and gradient of squared error with respect
    # to w (vector) for the given data X and y and the regularization parameter
    # lambda                                                                  

    # IMPLEMENT THIS METHOD 
    # error = (Y - X.W).T(Y - X.W) + Lambda.W.T.W
    # error_grad = X.T.X.W - X.T.Y + Lambda.W
    
    w = np.array(w).reshape(w.size, 1)
    xtw = np.dot(X,w)
    xtx = np.dot(X.T, X)
    # error = (Y - X.W).T(Y - X.W) + Lambda.W.T.W
    error = 0.5 * np.dot((xtw - y).T, (xtw - y)) + 0.5 * lambd * np.dot(w.T, w)
    # error_grad = X.T.X.W - X.T.Y + Lambda.W
    error_grad = np.dot(xtx, w) - np.dot(X.T, y) + lambd * w
    return error, error_grad.flatten()

def mapNonLinear(x,p):
    # Inputs:                                                                  
    # x - a single column vector (N x 1)                                       
    # p - integer (>= 0)                                                       
    # Outputs:                                                                 
    # Xp - (N x (p+1)) 
	
    # IMPLEMENT THIS METHOD
    Xp = np.ones((x.shape[0], p+1))
    for i in range(1, p+1):
        Xp[:, i] = x**i
    return Xp

# Main script

# Problem 1
# load the sample data                                                                 
if sys.version_info.major == 2:
    X,y,Xtest,ytest = pickle.load(open('sample.pickle','rb'))
else:
    X,y,Xtest,ytest = pickle.load(open('sample.pickle','rb'),encoding = 'latin1')

# LDA
means,covmat = ldaLearn(X,y)
ldaacc,ldares = ldaTest(means,covmat,Xtest,ytest)
print('LDA Accuracy = '+str(ldaacc))
# QDA
#means,covmats = qdaLearn(X,y)
#qdaacc,qdares = qdaTest(means,covmats,Xtest,ytest)
#print('QDA Accuracy = '+str(qdaacc))

# plotting boundaries
x1 = np.linspace(-5,20,100)
x2 = np.linspace(-5,20,100)
xx1,xx2 = np.meshgrid(x1,x2)
xx = np.zeros((x1.shape[0]*x2.shape[0],2))
xx[:,0] = xx1.ravel()
xx[:,1] = xx2.ravel()

fig = plt.figure(figsize=[12,6])
plt.subplot(1, 2, 1)

zacc,zldares = ldaTest(means,covmat,xx,np.zeros((xx.shape[0],1)))
plt.contourf(x1,x2,zldares.reshape((x1.shape[0],x2.shape[0])),alpha=0.3)
plt.scatter(Xtest[:,0],Xtest[:,1],c=ytest.ravel())
plt.title('LDA')

plt.subplot(1, 2, 2)

#zacc,zqdares = qdaTest(means,covmats,xx,np.zeros((xx.shape[0],1)))
#plt.contourf(x1,x2,zqdares.reshape((x1.shape[0],x2.shape[0])),alpha=0.3)
#plt.scatter(Xtest[:,0],Xtest[:,1],c=ytest.ravel())
#plt.title('QDA')

plt.show()
# Problem 2
if sys.version_info.major == 2:
    X,y,Xtest,ytest = pickle.load(open('diabetes.pickle','rb'))
else:
    X,y,Xtest,ytest = pickle.load(open('diabetes.pickle','rb'),encoding = 'latin1')

# add intercept
X_i = np.concatenate((np.ones((X.shape[0],1)), X), axis=1)
Xtest_i = np.concatenate((np.ones((Xtest.shape[0],1)), Xtest), axis=1)

w = learnOLERegression(X,y)
mle = testOLERegression(w,Xtest,ytest)

w_train = learnOLERegression(Xtest,ytest)
mle_train = testOLERegression(w_train,Xtest,ytest)

w_i = learnOLERegression(X_i,y)
mle_i = testOLERegression(w_i,Xtest_i,ytest)

w_i_Train = learnOLERegression(Xtest_i,ytest)
mle_Traini = testOLERegression(w_i_Train,Xtest_i,ytest)

print('MSE without intercept for Test '+str(mle))
print('MSE with intercept for Test'+str(mle_i))

print('MSE without intercept for Train '+str(mle_train))
print('MSE with intercept for Train '+str(mle_Traini))

# Problem 3
k = 101
lambdas = np.linspace(0, 1, num=k)
i = 0
mses3_train = np.zeros((k,1))
mses3 = np.zeros((k,1))
for lambd in lambdas:
    w_l = learnRidgeRegression(X_i,y,lambd)
    mses3_train[i] = testOLERegression(w_l,X_i,y)
    mses3[i] = testOLERegression(w_l,Xtest_i,ytest)
    i = i + 1
fig = plt.figure(figsize=[12,6])
plt.subplot(1, 2, 1)
plt.plot(lambdas,mses3_train)
print('<----MSE Ridge Train----->')
print(mses3_train)
print('<------End MSE Ridge Train------->')
plt.title('MSE for Train Data')
plt.subplot(1, 2, 2)
plt.plot(lambdas,mses3)
print('<---------MSE Ridge Test---------->')
print(mses3)
print('<------END Ridge Test------>')
plt.title('MSE for Test Data')

plt.show()

plt.show()
# Problem 4
k = 101
lambdas = np.linspace(0, 1, num=k)
i = 0
mses4_train = np.zeros((k,1))
mses4 = np.zeros((k,1))
opts = {'maxiter' : 20}    # Preferred value.                                                
w_init = np.ones((X_i.shape[1],1))
for lambd in lambdas:
    args = (X_i, y, lambd)
    w_l = minimize(regressionObjVal, w_init, jac=True, args=args,method='CG', options=opts)
    w_l = np.transpose(np.array(w_l.x))
    w_l = np.reshape(w_l,[len(w_l),1])
    mses4_train[i] = testOLERegression(w_l,X_i,y)
    mses4[i] = testOLERegression(w_l,Xtest_i,ytest)
    i = i + 1
fig = plt.figure(figsize=[12,6])
plt.subplot(1, 2, 1)
plt.plot(lambdas,mses4_train)
plt.plot(lambdas,mses3_train)
print("<------MSE Gradient Descent Train -------->")
print(mses4_train)
print("<---------MSE Gradient Descent Train End----------->")
plt.title('MSE for Train Data')
plt.legend(['Using scipy.minimize','Direct minimization'])

plt.subplot(1, 2, 2)
plt.plot(lambdas,mses4)
plt.plot(lambdas,mses3)
print("<-------MSE Gradient Descent Test Data------->")
print(mses4)
print("<-------MSE Gradient Descent Test End------->")
plt.title('MSE for Test Data')
plt.legend(['Using scipy.minimize','Direct minimization'])
plt.show()


# Problem 5
pmax = 7
lambda_opt = 0.7 # REPLACE THIS WITH lambda_opt estimated from Problem 3
mses5_train = np.zeros((pmax,2))
mses5 = np.zeros((pmax,2))
for p in range(pmax):
    Xd = mapNonLinear(X[:,2],p)
    Xdtest = mapNonLinear(Xtest[:,2],p)
    w_d1 = learnRidgeRegression(Xd,y,0)
    mses5_train[p,0] = testOLERegression(w_d1,Xd,y)
    mses5[p,0] = testOLERegression(w_d1,Xdtest,ytest)
    w_d2 = learnRidgeRegression(Xd,y,lambda_opt)
    mses5_train[p,1] = testOLERegression(w_d2,Xd,y)
    mses5[p,1] = testOLERegression(w_d2,Xdtest,ytest)
print("<-------MSE Non Linear train-------->")
print(mses5_train)
print("<-------MSE Non Linear train End-------->")
print("<-------MSE Non Linear test-------->")
print(mses5)
print("<-------MSE Non Linear test End-------->")

fig = plt.figure(figsize=[12,6])
plt.subplot(1, 2, 1)
plt.plot(range(pmax),mses5_train)
plt.title('MSE for Train Data')
plt.legend(('No Regularization','Regularization'))
plt.subplot(1, 2, 2)
plt.plot(range(pmax),mses5)
plt.title('MSE for Test Data')
plt.legend(('No Regularization','Regularization'))
plt.show()
