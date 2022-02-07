import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.optimize import minimize
from PIL import Image
import math
def pretreatment(ima):
    ima = ima.convert('L')         # 转化为灰度图像
    im = np.matrix(ima)        # 转化为二维数组
    im = im.astype(np.float)
    for i in range(im.shape[0]):    # 转化为二值矩阵
        for j in range(im.shape[1]):
            im[i, j] = math.fabs(float(im[i, j]) / 255.000-1)

    return im

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
    ax.matshow(image.reshape(20, 20).T, cmap='gray_r')
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
            ax_array[row][colomn].matshow(sample_images[10*row+colomn].reshape(20, 20).T, cmap='gray_r')
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


def predict_all(X, all_theta):
    h = sigmoid(np.dot(X, all_theta.T))  # (5000, 10)
    h_argmax = np.argmax(h, axis=1)
    h_argmax = h_argmax + 1
    return h_argmax


raw_X, raw_y = load_data('ex3data1.mat')
X = np.insert(raw_X, 0, 1, axis=1)
y = raw_y.flatten()
all_theta = one_vs_all(X, y, 1, 10)

#plot_an_image(raw_X, y)

'''y_pred = predict_all(X, all_theta)
accuracy = np.mean(y_pred==y)
print(accuracy)'''

ima=Image.open('D:\\python\\machine learning\\多元分类与前馈神经网络\\1.png') #读入图像
im=pretreatment(ima)  #调用图像预处理函数
im = np.insert(im, 0, 1)
h = sigmoid(np.dot(im.T.flatten(), all_theta.T))
h_argmax = np.argmax(h)
print(h_argmax+1, h.max())