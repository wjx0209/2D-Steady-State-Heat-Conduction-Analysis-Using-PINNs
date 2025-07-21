
import numpy as np
import matplotlib.pyplot as plt

# 参数定义
L = 1.0  # 板长 (m)
W = 1.0  # 板宽 (m)
k = 10.0  # 导热系数 (W/m·K)
q_s = 2000.0  # 热流密度 (W/m^2)
T1 = 30.0  # 边界温度 (°C)

# 解析解实现
def temperature_distribution(x, y, L, W, q_s, k, T1, terms=200):
    T = T1  # 起始温度为边界温度
    for n in range(1, terms + 1):
        # 解析解中的各项计算
        coefficient = (2 * (-1)**(n + 1) + 1) / (n**2 * np.pi**2 * np.cosh(n * np.pi * W / L))
        term = coefficient * np.sin(n * np.pi * x / L) * np.sinh(n * np.pi * y / L)
        T += (q_s * L / k) * term
    return T

# 创建网格
x = np.linspace(0, L, 100)
y = np.hstack([
    np.linspace(0, 0.1 * W, 50),  # 在 0 到 0.1W 的区域更密
    np.linspace(0.1 * W, W, 50)  # 在 0.1W 到 W 的区域较疏
])
X, Y = np.meshgrid(x, y)

# 计算温度分布
T = temperature_distribution(X, Y, L, W, q_s, k, T1)

# 绘制温度分布图
plt.figure(figsize=(6, 5))
contour = plt.contourf(X, Y, T, levels=50, cmap='jet')  # 使用 'jet' 配色方案
cbar = plt.colorbar(contour)
cbar.set_label('Temperature: T (°C)', fontsize=12)
cbar.ax.tick_params(labelsize=10)
plt.title('Temperature Distribution (Analytical)', fontsize=14)
plt.xlabel('x (m)', fontsize=12)
plt.ylabel('y (m)', fontsize=12)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.show()