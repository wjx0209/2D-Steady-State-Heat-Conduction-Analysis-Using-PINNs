
import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib import cm

# 参数定义
L = 1.0  # 板长
W = 1.0  # 板宽
M = 200  # x方向网格数
N = 200  # y方向网格数
q = 2000  # 热流密度
k = 10  # 导热系数
T_base = 30.0  # 基础温度
pi = math.pi

# 创建网格
x = np.linspace(0, L, M)
y = np.linspace(0, W, N)
X, Y = np.meshgrid(x, y)

# 展开项数
Nterm = 200

# 初始化温度矩阵
T = np.zeros((M, N))

# 解析解计算
for n in range(1, Nterm + 1):
    coefficient = (2 * ((-1) ** (n + 1) + 1)) / (n ** 2 * pi ** 2 * math.cosh(n * pi * W / L))
    term = (
        coefficient
        * np.sin(n * pi * X / L)
        * np.sinh(n * pi * Y / L)
    )
    T += (q * L / k) * term

# 加上基准温度
T += T_base

# 绘制温度分布图
plt.figure(figsize=(7, 6))
plt.title("Temperature Distribution, $N = 200$", fontname="serif")
contour = plt.contourf(X, Y, T, levels=100, cmap=cm.jet, extend="both")
plt.xlabel("$x/L$", fontname="serif")
plt.ylabel("$y/W$", fontname="serif")
cbar = plt.colorbar(contour)
cbar.set_label("Temperature: T (°C)", fontsize=12)
plt.tight_layout()
plt.savefig("plate.pdf")
plt.show()
