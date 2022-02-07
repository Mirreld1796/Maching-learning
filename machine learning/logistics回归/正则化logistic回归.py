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

def predict(theta, X):
    probability = sigmoid(X * theta.T)
    if probability >= 0.5:
        return 1, probability[0, 0]
    else:
        return 0, 1-probability[0, 0]


def feature_mapping(x1, x2, power):
    data = {}

    for i in np.arange(power + 1):
        for p in np.arange(i + 1):
            # 用pd.DataFrame()是字典的values必须为一维数组，而不能是矩阵
            data["f{}{}".format(i - p, p)] = np.array(np.multiply(np.power(x1, i - p), np.power(x2, p))).flatten()

#     data = {"f{}{}".format(i - p, p): np.array(np.multiply(np.power(x1, i - p), np.power(x2, p))).flatten()
#                 for i in np.arange(power + 1)
#                 for p in np.arange(i + 1)
#             }
    return pd.DataFrame(data)


def costReg(theta, X, y, l=1):
    # 不惩罚第一项
    theta = np.matrix(theta)  # 将theta转换为矩阵
    X = np.matrix(X)  # 将X转换为矩阵
    y = np.matrix(y)  # 将y转换为矩阵
    _theta = theta[0, 1:]
    reg = 1 / (2 * len(X)) * (_theta * _theta.T)[0, 0]
    return Compute_cost(theta, X, y) + reg


def gradientReg(theta, X, y, l=1):
    theta = np.matrix(theta)  # 将theta转换为矩阵
    X = np.matrix(X)  # 将X转换为矩阵
    y = np.matrix(y)  # 将y转换为矩阵
    reg = (1 / len(X)) * theta
    reg[0, 0] = 0
    return Gradient(theta, X, y) + reg


data = pd.read_csv('ex2data2.txt', names=['Test 1', 'Test 2', 'Accepted'])
positive = data[data['Accepted'].isin([1])]
negative = data[data['Accepted'].isin([0])]

x1 = np.matrix(data['Test 1'])
x2 = np.matrix(data['Test 2'])
_data2 = feature_mapping(x1, x2, power=6)

X = np.matrix(_data2)
y = np.matrix(data['Accepted']).T
theta = np.matrix(np.zeros(X.shape[1]))
#result2 = opt.fmin_tnc(func=costReg, x0=theta, fprime=gradientReg, args=(X, y, 2))
result2 = opt.minimize(fun=costReg, x0=theta, args=(X, y, 2), method='TNC', jac=gradientReg)
final_theta = result2['x']

fig, ax = plt.subplots(figsize=(10, 6.18))
ax.scatter(positive['Test 1'], positive['Test 2'], color='b', label='Accepted')
ax.scatter(negative['Test 1'], negative['Test 2'], s=50, color='r', marker='x', label='Not Accepted')
# 设置图例显示在图的上方
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width, box.height*0.8])
ax.legend(loc=1)
# 设置横纵坐标名
ax.set_xlabel('Exam 1 Score')
ax.set_ylabel('Exam 2 Score')
x = np.linspace(-1, 1.5, 250)
xx, yy = np.meshgrid(x, x)
# 由于250*250矩阵无法和theta相乘，先展开
z = np.matrix(feature_mapping(xx.ravel(), yy.ravel(), 6))
z = np.dot(z, final_theta)
# 再恢复
z = z.reshape(xx.shape)
plt.contour(xx, yy, z, 0)
plt.ylim(-0.8, 1.2)
#plt.show()
#p = np.matrix(feature_mapping(0.25, 0.1, 6))
#print(predict(np.matrix(final_theta), p))
