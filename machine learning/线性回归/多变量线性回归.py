import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

path = 'ex1data2.txt'
data2 = pd.read_csv(path, header=None, names=['Size', 'Bedrooms', 'Price'])
data2 = (data2 - data2.mean()) / data2.std()

# 计算代价函数
def Compute_cost(x, y, theta):
    inner = np.power((x * theta.T - y), 2)
    return np.sum(inner) / (2 * len(x))

# 批量梯度下降
def Gradient_descent(x, y, theta, alpha, epoch):
    cost = np.zeros(epoch)    # 初始化一个ndarray，包含每次epoch的cost
    m = x.shape[0]

    for i in range(epoch):
        theta = theta - (alpha / m) * (x * theta.T - y).T * x
        cost[i] = Compute_cost(x, y, theta)

    return theta, cost
# 正规矩阵求参数
def Normal_matrix(x,  y):
    theta = np.dot(np.dot(np.linalg.pinv(np.dot(x.T, x)), x.T), y).T
    return theta

data2.insert(0, 'Ones', 1)
cols = data2.shape[1]    # 列数
x = data2.iloc[:, 0:cols-1]
y = data2.iloc[:, cols-1:cols]
x = np.matrix(x.values)
y = np.matrix(y.values)
theta = np.matrix([[0, 0, 0]])
alpha = 1
epoch = 1000
final_theta, cost = Gradient_descent(x, y, theta, alpha, epoch)
fig, ax = plt.subplots(1, 2, figsize=(10, 6.18))


A = np.linspace(x[:, 1].min(), x[:, 1].max(), 80)
B = np.linspace(x[:, 2].min(), x[:, 2].max(), 80)
X, Y = np.meshgrid(A, B)
ax[0] = plt.axes(projection='3d')
Z = X / X * final_theta[0, 0] + X * final_theta[0, 1] + Y * final_theta[0, 2]
ax[0].scatter3D(x[:, 1], x[:, 2], y)

ax[0].set_xlabel('Size')
ax[0].set_ylabel('Bedrooms')
ax[0].set_zlabel('Price')
ax[0].plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='rainbow', edgecolor='none')

"""
ax[1].plot(np.arange(epoch), cost, 'r')
ax[1].set_xlabel("Iterations") 
ax[1].set_ylabel("Cost")
ax[1].set_title("Error vs. Training Epoch")
"""
plt.show()
print(final_theta)
print(Normal_matrix(x, y))

