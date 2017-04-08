"""
This is an implementation of logistic regression in python
This script contains two parts:
1.maximum likelihood estimation
2.gradient ascent
"""

import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.cross_validation import train_test_split


def logistic_regression(X, Y, theta, alpha, max_iters):
    """
    use logistic regression to fit a model
    :param X: input features
    :param Y: output values
    :param theta: parameters to be fit
    :param alpha: learning rate
    :return: final theta
    """
    ##write a loop till convergence
    for p in range(max_iters):
        new_theta=gradient_descent(X,Y,theta,alpha)
        if new_theta==theta:
            break
        theta=new_theta
        #print ('theta is {0}'.format(theta))
    return new_theta


def gradient_descent(X, Y, theta, alpha):
    grad = []
    for j in range(len(theta)):
        elem= theta[j] + alpha * grad_j(X, Y, theta, alpha, j)
        grad.append(elem)
    #print('gradient this time is {0}'.format(grad))
    return grad


def grad_j(X, Y, theta, alpha, j):
    """
    calculte gradient
    :param X: input features
    :param Y: output values
    :param theta: parameters to be fit
    :param alpha:learning rate
    :param i:index of the example
    :return: grad_j
    """
    m = len(Y)
    #print (' m is {0}'.format(m))
    grad_j = 0
    for i in range(m):
        #print ('theta is {0},X[i] is {1},h() is {2}'.format(theta, X[i],h(X[i], Y[i], theta)))
        grad_j += (((Y[i] - h(X[i], Y[i], theta))*X[i][j]))
    return grad_j


def h(x, y, theta):
    """
    calculate prob
    :param x: input list
    :param y: output value
    :param theta: parameter list
    :return: value of cost
    """
    z = hypo(x, theta)
    return signoid(z)


def signoid(z):
    """
    conditional probability distribution
    :param z:input value
    :return: odds that this y occurs given this x (float)
    """
    return 1 / (1 + np.exp(-1 * z))


def hypo(x, theta):
    """
    calculate linear part
    :param x: inout features of one example
    :param theta: parameters to be fit
    :return: z value
    """
    n = len(x)
    z = 0
    for i in range(n):
        z += (x[i] * theta[i])

    return z


def read_data():
    """
    read X,Y from files
    :return: X,Y
    """
    """
    df = pd.read_csv('data.csv', header=0)
    df.columns = ['grade1', 'grade2', 'label']
    df['0'] = 1
    #print (df)
    return (df[['0', 'grade1', 'grade2']], df[['label']])
    """
    X = pd.read_table('logistic_x.txt', sep=' +', header=None)
    Y = pd.read_table('logistic_y.txt', sep=' +', header=None)
    X.columns=['x1','x2']
    X['x0']=1.
    return X[['x0','x1','x2']],Y



def clean_data(X, Y):
    """
    clean and normalize X,Y
    :param X: input features pandas dataframe
    :param Y: output values pandas dataframe
    :return: cleaned X,Y in numpy array form
    """
    # normalize X
    X = normalize(X)
    X=np.array(X)
    # clean Y
    #Y = Y['label'].map(lambda x: float(x.rstrip(';')))

    Y=Y[0].map(lambda y:(y+1)/2.)
    Y = np.array(Y)

    return X, Y


def normalize(X):
    """
    project data from a wide range to a smaller range like (0,1)
    :param X: dataframe to be processed
    :return: normlized X(numpy.ndarray)
    """

    scaler = preprocessing.MinMaxScaler(feature_range=(-1, 1))
    X = np.array(X)
    X = scaler.fit_transform(X)
    return X


def test_data(test_X, test_Y, theta):
    """
    test this model
    :param test_X: test set of X
    :param test_Y: test set of Y
    :param theta: fitted theta
    :return: score
    """
    count=0
    for i in range(len(test_Y)):
        y_expected=round(h(test_X[i],test_Y[i],theta))
        #print ('raw y is {0}, y expected is {1}, y given is {2}'.format(h(X[i],Y[i],theta),y_expected,Y[i]))
        if y_expected==test_Y[i]:
            count+=1
    return count/float(len(test_Y))


# read and clean data
(X, Y) = read_data()
X, Y = clean_data(X, Y)
print (X)
print (Y)
# divide data set into training and testing set
training_X, testing_X, training_Y, testing_Y = train_test_split(X, Y, test_size=0.33)
#print ('training_X is {0}'.format(training_X))
#print ('training_y is {0}'.format(training_Y))
#print ('testing_Y is {0}'.format(testing_Y))
# ('grad_j is ')
#print (grad_j(training_X, training_Y, [0, 1, 2], 0.1, 0))
#print(
#'first gradient descent gives out:\n theta is {0}'.
#    format(gradient_descent(training_X, training_Y, [0, 0, 0], 0.1, )))

# logistic regression
theta=logistic_regression(training_X,training_Y,[0,0,0],0.1,1000)
print('The final theta is {0}'.format(theta))

# test data

error=test_data(testing_X,testing_Y,theta)
print ('About {0:.2f}% data tested was correct!'.format(error*100))

