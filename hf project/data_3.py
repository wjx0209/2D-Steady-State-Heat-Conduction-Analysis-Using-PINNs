
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange

# 检查Eager模式
print("TensorFlow version:", tf.__version__)
print("Eager Execution:", tf.executing_eagerly()) # 在TF 2.x下通常应为 True

L = 1.0
W = 1.0
k_original = 10.0
q_s_original = 2000.0
T1_original = 30.0

k = 1.0
q_s = 1.0 / L
T1 = 0.0

layers = [2, 128, 128, 128, 1]

N_int = 2000
x_int = np.random.rand(N_int,1)*L
y_int = np.random.rand(N_int,1)*W
X_int = np.hstack((x_int,y_int))

N_bc = 200
y_left = np.random.rand(N_bc,1)*W
x_left = np.zeros_like(y_left)
X_left = np.hstack((x_left,y_left))

y_right = np.random.rand(N_bc,1)*W
x_right = L*np.ones_like(y_right)
X_right = np.hstack((x_right,y_right))

x_bottom = np.random.rand(N_bc,1)*L
y_bottom = np.zeros_like(x_bottom)
X_bottom = np.hstack((x_bottom,y_bottom))

x_top = np.random.rand(N_bc,1)*L
y_top = W*np.ones_like(x_top)
X_top = np.hstack((x_top,y_top))

X_int_tf = tf.convert_to_tensor(X_int, dtype=tf.float32)
X_left_tf = tf.convert_to_tensor(X_left, dtype=tf.float32)
X_right_tf = tf.convert_to_tensor(X_right, dtype=tf.float32)
X_bottom_tf = tf.convert_to_tensor(X_bottom, dtype=tf.float32)
X_top_tf = tf.convert_to_tensor(X_top, dtype=tf.float32)

def neural_net(layers):
    model = tf.keras.Sequential()
    for i in range(len(layers)-1):
        if i < len(layers)-2:
            model.add(tf.keras.layers.Dense(layers[i+1], activation='tanh', 
                                            kernel_initializer='glorot_normal'))
        else:
            model.add(tf.keras.layers.Dense(layers[i+1], activation=None, 
                                            kernel_initializer='glorot_normal'))
    return model

model = neural_net(layers)

def T_model(x, y):
    X = tf.concat([x, y], axis=1)
    return model(X)

def pde_residual(x, y):
    with tf.GradientTape(persistent=True) as tape2:
        tape2.watch(x)
        tape2.watch(y)
        with tf.GradientTape(persistent=True) as tape1:
            tape1.watch(x)
            tape1.watch(y)
            T_pred = T_model(x, y)
        T_x = tape1.gradient(T_pred, x)
        T_y = tape1.gradient(T_pred, y)
    T_xx = tape2.gradient(T_x, x)
    T_yy = tape2.gradient(T_y, y)
    del tape1, tape2
    return T_xx + T_yy

def loss_fn():
    x_int_tf_ = tf.reshape(X_int_tf[:,0], (-1,1))
    y_int_tf_ = tf.reshape(X_int_tf[:,1], (-1,1))
    res = pde_residual(x_int_tf_, y_int_tf_)
    loss_pde = tf.reduce_mean(tf.square(res))

    T_left = T_model(X_left_tf[:,0:1], X_left_tf[:,1:2])
    loss_left = tf.reduce_mean(tf.square(T_left - T1))

    T_right = T_model(X_right_tf[:,0:1], X_right_tf[:,1:2])
    loss_right = tf.reduce_mean(tf.square(T_right - T1))

    T_bottom = T_model(X_bottom_tf[:,0:1], X_bottom_tf[:,1:2])
    loss_bottom = tf.reduce_mean(tf.square(T_bottom - T1))

    x_top_var = X_top_tf[:,0:1]
    y_top_var = X_top_tf[:,1:2]
    with tf.GradientTape() as tape:
        tape.watch(y_top_var)
        T_top = T_model(x_top_var, y_top_var)
    dT_dy_top = tape.gradient(T_top, y_top_var)
    loss_top = tf.reduce_mean(tf.square(dT_dy_top - (q_s/k)))

    return loss_pde, loss_left, loss_right, loss_bottom, loss_top

optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)

epochs = 5000

# tbar = trange(epochs)
for epoch in trange(epochs):
    with tf.GradientTape() as tape:
        loss_pde, loss_left, loss_right, loss_bottom, loss_top = loss_fn()
        loss_value = loss_pde + loss_left + loss_right + loss_bottom + loss_top
    grads = tape.gradient(loss_value, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    if epoch % 100 == 0:
        print(f"Epoch: {epoch}, Loss PDE: {loss_pde.numpy()}, Loss BC: {(loss_left+loss_right+loss_bottom+loss_top).numpy()}")

nx, ny = 100, 100
x_plot = np.linspace(0, L, nx)
y_plot = np.linspace(0, W, ny)
X_plot, Y_plot = np.meshgrid(x_plot, y_plot)
X_plot_tf = tf.convert_to_tensor(X_plot.reshape(-1,1), dtype=tf.float32)
Y_plot_tf = tf.convert_to_tensor(Y_plot.reshape(-1,1), dtype=tf.float32)

T_pred = T_model(X_plot_tf, Y_plot_tf).numpy().reshape(ny,nx)
T_actual = T_pred * (q_s_original * L / k_original) + T1_original

plt.figure(figsize=(6,5))
contour = plt.contourf(X_plot, Y_plot, T_actual, levels=50, cmap='jet')
cbar = plt.colorbar(contour)
cbar.set_label('Temperature (°C)', fontsize=12)
plt.title('Temperature Distribution (PINN)', fontsize=14)
plt.xlabel('x (m)', fontsize=12)
plt.ylabel('y (m)', fontsize=12)
plt.show()
