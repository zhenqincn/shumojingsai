from scipy import optimize
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import transforms

x = np.array([i for i in range(2, 13)], dtype=np.float64)
y = [0.1038360595703125, 0.07921600341796875, 0.05683135986328125,
     0.038238525390625, 0.02326202392578125, 0.01242828369140625, 0.00612640380859375,
     0.0020751953125, 0.00083160400390625, 0.00019073486328125, 3.0517578125e-05]
z1 = np.polyfit(x, y, 2)#用3次多项式拟合
p1 = np.poly1d(z1)
print(p1) #在屏幕上打印拟合多项式


def func1(x):
    return np.array([0.02 for _ in range(len(x))])


def func2(x):
    return 0.001569*x*x - 0.0319 * x + 0.1601


def find_cure_intersects(x, y1, y2):  #计算两条曲线的交点
    d = y1-y2
    idx = np.where(d[:-1]*d[1:]<=0)[0]
    x1, x2 = x[idx], x[idx+1]
    d1, d2 = d[idx], d[idx+1]
    return -d1*(x2-x1)/(d2-d1) + x1


f1 = func1(x)
f2 = func2(x)
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(x, f1)
ax.plot(x, f2)

x1 = find_cure_intersects(x, f1, f2)

ax.plot(x1, func1(x1), "o")
ax.fill_between(x, f1, f2, where=f1 > f2, facecolor="green", alpha=0.5)

trans = transforms.blended_transform_factory(ax.transData, ax.transAxes)
a =ax.text(0.05, 0.95, u"直线和二次曲线的交点",
    transform=ax.transAxes,
    verticalalignment = "top",
    fontsize = 18,
    bbox={"facecolor":"red", "alpha":0.4, "pad":10})
arrow = {"arrowstyle":"fancy,tail_width=0.6",
          "facecolor":"gray",
          "connectionstyle":"arc3,rad=-0.3"}
ax.annotate(u"交点",
    xy=(x1, func1(x1)),xycoords="data",
    xytext=(0.05, 0.5), textcoords="axes fraction",
    arrowprops=arrow)

plt.show()
