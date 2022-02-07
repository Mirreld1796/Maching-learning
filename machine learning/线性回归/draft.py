import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import math
def pretreatment(ima):
    ima = ima.convert('L')         # 转化为灰度图像
    im = np.matrix(ima)        # 转化为二维数组
    im = im.astype(np.float64)
    for i in range(im.shape[0]):    # 转化为二值矩阵
        for j in range(im.shape[1]):
            im[i, j] = math.fabs(float(im[i, j]) / 255.000-1)

    return im


a = np.matrix([[1,2,3,4]])
print(a.reshape(2, 2))