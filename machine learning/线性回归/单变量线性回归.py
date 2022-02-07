import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

path = 'ex1data1.txt'
data = pd.read_csv(path, header=None, names=['Population', 'Profit'])
#data.plot(kind='scatter', x='Population', y='Profit', figsize=(10, 6.18))
#plt.show()

# 计算代价函数
def Compute_cost(x, y, theta):
    inner = np.power((x * theta.T - y), 2)
    return np.sum(inner) / (2 * len(x))

# 批量梯度下降
def Gradient_descent(x, y, theta, alpha, epoch):
    cost = np.zeros(epoch)    # 初始化一个ndarray，包含每次epoch的cost
    m = x.shape[0]

    for i in range(epoch):
        temp = theta - (alpha / m) * (x * theta.T - y).T * x
        theta = temp
        cost[i] = Compute_cost(x, y, theta)

    return theta, cost

data.insert(0, 'Ones', 1)
cols = data.shape[1]    # 列数
x = data.iloc[:, 0:cols-1]
y = data.iloc[:, cols-1:cols]
x = np.matrix(x.values)
y = np.matrix(y.values)
theta = np.matrix([0, 0])
alpha = 0.01
epoch = 1000
final_theta, cost = Gradient_descent(x, y, theta, alpha, epoch)
x_axis = np.linspace(data['Population'].min(), data['Population'].max(), 100)
y_axis = final_theta[0, 0] + final_theta[0, 1] * x_axis
fig, axes = plt.subplots(1, 2, figsize=(20, 6.18))
axes[0].plot(x_axis, y_axis, 'r', label='Prediction')
axes[0].scatter(data['Population'], data['Profit'], label='Tracing data')
axes[0].legend(loc=2)
axes[0].set_xlabel("Population")
axes[0].set_ylabel("Profit")
axes[0].set_title("Predicted Profit vs. Population Size")
axes[1].plot(np.arange(epoch), cost, color='r')
axes[1].set_xlabel("Iterations")
axes[1].set_ylabel("Cost")
axes[1].set_title("Error vs. Training Epoch")
plt.show()
#print(final_theta)






