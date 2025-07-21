
import numpy as np
import matplotlib.pyplot as plt

# 参数定义
L = 1.0   # 板长 (m)
W = 1.0   # 板宽 (m)
k = 10.0  # 导热系数 (W/m·K)
q_s = 2000.0  # 热流密度 (W/m^2)
T1 = 30.0     # 边界温度 (°C)

# 网格划分
Nx = 51
Ny = 51
dx = L/(Nx-1)
dy = W/(Ny-1)

# 初始化温度场
T = np.zeros((Nx, Ny)) + T1

# 设置Dirichlet边界条件（左、右、下）
T[0, :] = T1    # 左边界
T[-1, :] = T1   # 右边界
T[:, 0] = T1    # 下边界

# 收敛标准
tol = 1e-5
max_iter = 100000
error = 1.0
iter_count = 0

# 如果q_s向下，则q_s在坐标系中是负方向热流：q_s = -2000 W/m²
# q_s = -k dT/dy => -2000 = -10 dT/dy => dT/dy = 200 °C/m
# 顶部边界条件: T[i,Ny-1] = T[i,Ny-2] + 200*dy

while error > tol and iter_count < max_iter:
    error = 0.0
    T_old = T.copy()
    
    # 应用顶部Neumann边界条件
    for i in range(Nx):
        T[i, Ny-1] = T[i, Ny-2] + 200.0 * dy  # 修改为加号

    # 内点更新（Gauss-Seidel迭代）
    for i in range(1, Nx-1):
        for j in range(1, Ny-1):
            # 顶部已由Neumann设置，这里仍可更新内部点
            # 但要注意顶部最上一行由Neumann给定不需更新
            if j < Ny-1:
                T[i,j] = 0.25*(T[i+1,j] + T[i-1,j] + T[i,j+1] + T[i,j-1])

    # 计算误差
    error = np.max(np.abs(T - T_old))
    iter_count += 1

print("迭代次数:", iter_count)
print("最大误差:", error)

# 绘图
x = np.linspace(0, L, Nx)
y = np.linspace(0, W, Ny)
X, Y = np.meshgrid(x, y, indexing='ij')

plt.figure(figsize=(6,5))
contour = plt.contourf(X, Y, T, 50, cmap='jet')
cbar = plt.colorbar(contour)
cbar.set_label('Temperature: T (°C)', fontsize=12)
plt.title('Temperature Distribution (Numerical)', fontsize=14)
plt.xlabel('x (m)', fontsize=12)
plt.ylabel('y (m)', fontsize=12)
plt.show()
