import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.optimize as opt

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def Compute_cost(theta, X, y):
    theta = np.matrix(theta)  # 将theta转换为矩阵
    X = np.matrix(X)  # 将X转换为矩阵
    y = np.matrix(y)  # 将y转换为矩阵
    first = np.multiply((-y), np.log(sigmoid(X * theta.T)+1e-6))
    second = np.multiply((1 - y), np.log(1 - sigmoid(X * theta.T)+1e-6))
    return np.sum(first - second) / (len(X))


def Gradient(theta, X, y):
    theta = np.matrix(theta)  # 将theta转换为矩阵
    X = np.matrix(X)  # 将X转换为矩阵
    y = np.matrix(y)  # 将y转换为矩阵
    return (sigmoid(X * theta.T) - y).T * X / len(X)


def Gradient_descent(theta, X, y, alpha, epoch):
    cost = np.zeros(epoch)
    m = X.shape[0]

    for i in range(epoch):
        theta = theta - (alpha / m) * (sigmoid(X * theta.T) - y).T * X
        cost[i] = Compute_cost(theta, X, y)
    return theta, cost


def predict(theta, X):
    probability = sigmoid(X * theta.T)
    if probability >= 0.5:
        return 1, probability[0, 0]
    else:
        return 0, 1-probability[0, 0]

data = pd.read_csv('ex2data1.txt', names=['exam1', 'exam2', 'admitted'])
positive = data[data['admitted'].isin([1])]
negative = data[data['admitted'].isin([0])]

data.insert(0, 'Ones', 1)
X = np.matrix(data.iloc[:, 0:3])
y = np.matrix(data.iloc[:, 3:4])
theta = np.matrix(np.zeros(X.shape[1]))
alpha = 0.004
epoch = 200000
#final_theta, cost = Gradient_descent(theta, X, y, alpha, epoch)
#result1 = opt.fmin_tnc(func=Compute_cost, x0=theta, fprime=Gradient, args=(X, y))
result2 = opt.minimize(fun=Compute_cost, x0=theta, args=(X, y), method='TNC', jac=Gradient)
#print(result2)
theta_final = np.matrix(result2['x'])
#pp = np.matrix([1, 47, 32])
#print(predict(theta_final, pp))

fig, ax = plt.subplots(figsize=(10, 6.18))
ax.scatter(positive['exam1'], positive['exam2'], color='b', label='admitted')
ax.scatter(negative['exam1'], negative['exam2'], s=50, color='r', marker='x', label='Not admitted')
# 设置图例显示在图的上方
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width, box.height*0.8])
ax.legend(loc=1)
# 设置横纵坐标名
ax.set_xlabel('Exam 1 Score')
ax.set_ylabel('Exam 2 Score')
x1 = np.linspace(30, 105, 100)
x2 = -(theta_final[0, 0] + x1 * theta_final[0, 1]) / theta_final[0, 2]
ax.plot(x1, x2)
ax.set_xlabel('Exam1')
ax.set_ylabel('Exam2')
ax.set_title('Disition Boundary')

plt.show()


