import datetime

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

import pandas as pd

ok_data_filepath = 'E:\\workspace\\BackTesting\\back_testing_result\\back_testing_20210202_triMa\\min1.csv'
df = pd.read_csv(ok_data_filepath)
# 对时间进行转换  # todo 可能还有其他要转换
df['datetime'] = df['datetime'].apply(
    lambda x: datetime.datetime.strptime(x, '%Y-%m-%d %H:%M:%S'))



def test_plat():
    print(df)
    # 画图
    fig = plt.figure()
    ax1 = fig.gca(projection='3d')
    # ax.plot_trisurf(df.datetime, df.Open, df.boll_mid, cmap=plt.cm.Spectral, linewidth=0.2)
    # # 调整角度，第一个数字为上下，第二个数字为左右。
    # ax.view_init(30, 80)
    # plt.show()

    z = np.linspace(0, 13, 1000)
    x = 5 * np.sin(z)
    y = 5 * np.cos(z)
    zd = 13 * np.random.random(100)
    xd = 5 * np.sin(zd)
    yd = 5 * np.cos(zd)
    ax1.scatter3D(xd, yd, zd, cmap='Blues')  # 绘制散点图
    ax1.plot3D(x, y, z, 'gray')  # 绘制空间曲线
    plt.show()


def test_plot():
    print(df)
    # 画图
    fig = plt.figure()
    ax3 = fig.gca(projection='3d')
    # ax.plot_trisurf(df.datetime, df.Open, df.boll_mid, cmap=plt.cm.Spectral, linewidth=0.2)
    # 定义三维数据
    xx = np.arange(-5, 5, 0.5)
    yy = np.arange(-5, 5, 0.5)
    X, Y = np.meshgrid(xx, yy)
    Z = np.sin(X) + np.cos(Y)
    dt1 = pd.date_range(start="20190101", end="20190331", freq="D")  # freq="D"表示频率为每一天
    dt1 = list(dt1[:len(xx)])
    print(type(dt1), dt1)

    # 作图
    ax3.plot_surface(X, Y, dt1, cmap='rainbow')
    # ax3.contour(X,Y,Z, zdim='z',offset=-2，cmap='rainbow)   #等高线图，要设置offset，为Z的最小值
    plt.show()