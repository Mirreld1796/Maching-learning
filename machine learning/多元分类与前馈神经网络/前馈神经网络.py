import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.optimize import minimize


def load_data(path):
    data = loadmat(path)
    X = data['X']
    y = data['y']
    return X, y

def plot_an_image(X, y):
    """随机打印一个数字"""
    pick_one = np.random.randint(0, 5000)
    image = X[pick_one, :]
    fig, ax = plt.subplots(figsize=(1, 1))
    ax.matshow(image.reshape(20, 20), cmap='gray_r')
    plt.xticks([])
    plt.yticks([])
    plt.show()
    print(y[[pick_one]])

def plot_100_images(X, y):
    sample_idx = np.random.choice(np.arange(X.shape[0]), 100) #随机选100个样本
    sample_images = X[sample_idx, :]

    fig, ax_array = plt.subplots(10, 10, sharey=True, sharex=True, figsize=(8, 8))
    for row in range(10):
        for colomn in range(10):
            ax_array[row][colomn].matshow(sample_images[10*row+colomn].reshape(20, 20), cmap='gray_r')
    plt.xticks([])
    plt.yticks([])
    plt.show()


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def regularized_cost(theta, X, y, l=1):
    theta = np.matrix(theta)  # 将theta转换为矩阵
    X = np.matrix(X)  # 将X转换为矩阵
    y = np.matrix(y)  # 将y转换为矩阵
    thetaReg = theta[0, 1:]
    first = np.multiply((-y), np.log(sigmoid(X * theta.T)))
    second = np.multiply((1 - y), np.log(1 - sigmoid(X * theta.T)))
    Reg = 1 / (2 * len(X)) * (thetaReg * thetaReg.T)[0, 0]
    return (first - second).mean() + Reg


def regularized_gradient(theta, X, y, l=1):
    theta = np.matrix(theta)  # 将theta转换为矩阵
    X = np.matrix(X)  # 将X转换为矩阵
    y = np.matrix(y)  # 将y转换为矩阵
    first = (sigmoid(X * theta.T) - y).T * X / len(X)
    Reg = theta / len(X)
    Reg[0, 0] = 0
    return first + Reg


def one_vs_all(X, y, l, K):
    # K代表有多少种标签，即有多少种数字，10种
    all_theta = np.zeros((K, X.shape[1]))  # (10, 400)
    X = np.matrix(X)  # 将X转换为矩阵
    y = np.matrix(y)  # 将y转换为矩阵
    for i in range(1, K+1):
        theta = np.matrix(np.zeros(X.shape[1]))
        y_i = np.matrix([1 if label == i else 0 for label in np.array(y).flatten()]).T  # 这里必须要转置
        ret = minimize(fun=regularized_cost, x0=theta, args=(X, y_i, l), method='TNC',
                       jac=regularized_gradient, options={'disp': True})
        all_theta[i-1, :] = ret['x']
    return all_theta


def load_weight(path):
    data = loadmat(path)
    return data['Theta1'], data['Theta2']

theta1, theta2 = load_weight('ex3weights.mat')
X, y = load_data('ex3data1.mat')
y = y.flatten()
X = np.insert(X, 0, 1, axis=1)
a1 = X
z2 = np.dot(a1, theta1.T)
z2 = np.insert(z2, 0, 1, axis=1)
a2 = sigmoid(z2)
z3 = np.dot(a2, theta2.T)
a3 = sigmoid(z3)

y_pred = np.argmax(a3, axis=1) + 1
accuracy = np.mean(y_pred == y)
print(accuracy)








