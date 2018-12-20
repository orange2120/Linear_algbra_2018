import pandas as pd
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

attrs = ['AMB', 'CH4', 'CO', 'NMHC', 'NO', 'NO2',
        'NOx', 'O3', 'PM10', 'PM2.5', 'RAINFALL', 'RH',
        'SO2', 'THC', 'WD_HR', 'WIND_DIR', 'WIND_SPEED', 'WS_HR']
DAYS = np.array([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])

def read_TrainData(filename, N):
    #N: how many hours to be as inputs
    raw_data = pd.read_csv(filename).as_matrix()
    # 12 months, 20 days per month, 18 features per day. shape: (4320 , 24)
    data = raw_data[:, 3:] #first 3 columns are not data
    data = data.astype('float')
    X, Y = [], []
    for i in range(0, data.shape[0], 18*20):
        # i: start of each month
        days = np.vsplit(data[i:i+18*20], 20) # shape: 20 * (18, 24)
        concat = np.concatenate(days, axis=1) # shape: (18 feat, 480(day*hr))
        # take every N hours as x and N+1 hour as y
        for j in range(0, concat.shape[1]-N):
            features = concat[:, j:j+N].flatten() #the data of previous N hours
            features = np.append(features, [1]) # add w0
            X.append(features)
            Y.append([concat[9, j+N]]) #9th feature is PM2.5
    X = np.array(X)
    Y = np.array(Y)
    return X, Y

#from 1/23 0am, 1am ..23pm... 2/23, 0am, .... ~ 12/31 23p.m, total 2424 hours
#will give you a matrix 2424 * (18*N features you need)
def read_TestData(filename, N):
	#only handle N <= 48(2 days)
    assert N <= 48
    raw_data = pd.read_csv(filename).as_matrix()
    data = raw_data[:, 3:]
    data = data.astype('float')
    surplus = DAYS - 20 #remaining days in each month after 20th
    test_X = []
    test_Y = [] #ground truth
    for i in range(12): # 12 month
        # i: start of each month
        start = sum(surplus[:i])*18
        end = sum(surplus[:i+1])*18
        days = np.vsplit(data[start:end], surplus[i])
        concat = np.concatenate(days, axis=1) # shape: (18 feat, (day*hr))
        for j in range(48, concat.shape[1]): #every month starts from 23th
            features = concat[:, j-N:j].flatten()
            features = np.append(features, [1]) # add w0
            test_X.append(features)
            test_Y.append([concat[9, j]])
    test_X = np.array(test_X)
    test_Y = np.array(test_Y)
    return test_X, test_Y


class Linear_Regression(object):
    def __init__(self):
        pass
    def train(self, train_X, train_Y):
        #TODO
        W = np.dot(np.dot(np.linalg.inv(np.dot(np.transpose(train_X), train_X)), np.transpose(train_X)), train_Y)
        #np.savetxt("W.txt", W)
        self.W = W #save W for later prediction

    def predict(self, test_X):
        #TODO
        #predict_Y = ...?
        predict_Y = np.dot(test_X, self.W)
        return predict_Y

    def predictTY(self, train_X):
        return np.dot(train_X ,self.W)

def gradDes(train_X, train_Y, theta, alpha, num_iter):
    m = len(train_Y)
    n = len(theta)
    tmp = np.zeros((n, 1))
    for iter in range(1, num_iter + 1):
        for initer in range(0, n):
            tmp[initer] = theta[initer] - alpha * (1.0 / m) * sum(np.transpose(np.dot( theta - train_Y, train_X)))
        theta = tmp
    return theta

def gradient_descent_2(alpha, x, y, numIterations):
    m = x.shape[0] # number of samples
    theta = np.ones(109)
    x_transpose = x.transpose()
    for iter in range(0, numIterations):
        hypothesis = np.dot(x, theta)
        loss = hypothesis - y
        J = np.sum(loss ** 2) / (2 * m)  # cost     
        gradient = np.dot(x_transpose, loss) / m         
        theta = theta - alpha * gradient  # update
    return theta

def gradientDescent(x, y, theta, alpha, m, numIterations):
    xTrans = x.transpose()
    #loss = np.ndarray((), dtype = float)
    for i in range(0, numIterations):
        hypothesis = np.dot(x, theta)
        for j in range(len(theta)):
            loss.append(loss, hypothesis[j] - y ) 
        np.savetxt("./loss.txt", loss)

        print("hypothesis")
        print(hypothesis.shape)
        print("loss")
        print(loss.shape)
        print("theta")
        print(theta.shape)

        # avg cost per example (the 2 in 2*m doesn't really matter here.
        # But to be consistent with the gradient, I include it)
        cost = np.sum(loss ** 2) / (2 * m)
        print("Iteration %d | Cost: %f" % (i, cost))
        # avg gradient per example
        gradient = np.dot(xTrans, loss) / m
        
        print(gradient.shape)
        print(type(gradient))
        # update
        for j in range(len(theta)):
            theta[j] = theta[j] - alpha * gradient[j]
    return theta


def genData(numPoints, bias, variance):
    x = np.zeros(shape=(numPoints, 2))
    y = np.zeros(shape=numPoints)
    # basically a straight line
    for i in range(0, numPoints):
        # bias feature
        x[i][0] = 1
        x[i][1] = i
        # our target variable
        y[i] = (i + bias) + random.uniform(0, 1) * variance
    return x, y

def MSE(predict_Y, real_Y):
    #TODO :mean square error
    s = float(0)
    for i in range(len(predict_Y)):
        s += (predict_Y[i]-real_Y[i])**2
    loss = s/len(predict_Y)

    # loss = ?
    return loss

def plot_range(begin, end):
    trl = []
    tel = []
    for i in range(begin, end):
        print(i)
        train_X, train_Y = read_TrainData('train.csv', N=i)
        test_X, test_Y = read_TestData('test.csv', N=i)
        m = Linear_Regression()
        m.train(train_X, train_Y)
        predict_Y = m.predict(test_X)
        trl.append(MSE(train_Y, m.predictTY(train_X)))
        tel.append(MSE(predict_Y, test_Y))
    plot(trl, tel)

def plot(tr_set_loss, te_set_loss, path = './plot.png'):
    assert len(tr_set_loss) == len(te_set_loss)
    length = len(tr_set_loss)
    plt.plot(range(1, length+1), tr_set_loss, 'b', label='train loss')
    plt.plot(range(1, length+1), te_set_loss, 'r', label='test loss')
    plt.legend()
    plt.xlabel('N')
    plt.ylabel('MSE loss')
    plt.savefig(path)

if __name__ == '__main__' :
    N = 6
    train_X, train_Y = read_TrainData('train.csv', N=N)
    # train_X (5688, 109)
    # train_Y (5688, 1)
    model = Linear_Regression()
    model.train(train_X, train_Y)
    test_X, test_Y = read_TestData('test.csv', N=N)
    predict_Y = model.predict(test_X)
    test_loss = MSE(predict_Y, test_Y)

    #theta = np.zeros((train_X.shape[1],1))
    #theta = gradDes(train_X, train_Y, theta, 0.1, 3000)
    #predGrad_Y = np.dot(train_X, theta)
    theta = np.ones(train_X.shape[1])
    np.savetxt('theta0.txt' ,theta)
    m, n = np.shape(train_X)
    theta = gradientDescent(train_X, train_Y, theta, 0.01, m, 3000)
    predGrad_Y3 = np.dot(train_X, theta)
    np.savetxt('theta.txt' ,theta)
    gradLoss3 = MSE(predGrad_Y3, test_X)
    

    plot_range(1,48)

    print(test_loss)
    #print(gradLoss)
    #print(gradLoss2)
    print(gradLoss3)
