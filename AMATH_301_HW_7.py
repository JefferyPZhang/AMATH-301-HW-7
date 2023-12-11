import numpy as np
import matplotlib.pyplot as plt
import scipy
from matplotlib import cm

# Problem 1

x_i = 0
x_f = 0
t_f = 5

def solve_BVP(dt):
    t = np.arange(0, t_f + dt, dt)
    n = t.size
    v = -2 * np.ones(n - 2)
    u = np.ones(n - 3)
    A = (1 / dt ** 2) * (np.diag(v) + np.diag(u, 1) + np.diag(u, -1)) + np.eye(n - 2)
    b = 3 * np.sin(t[1 : -1])
    x_int = scipy.linalg.solve(A, b)
    x = np.zeros(n)
    x[0] = x_i
    x[1 : -1] = x_int.reshape(-1)
    x[-1] = x_f
    x_true = lambda t: -1.5 * (t * np.cos(t) - 5 * (np.cos(5) / np.sin(5)) * np.sin(t))
    return A, b, x, np.max(np.abs(x - x_true(t)))

A1, A2, A3, A4 = solve_BVP(0.1)
print(A3)
A5 = np.zeros(5)
A5[0] = A4
A6 = 2

for i in range (1, 5):
    dt = 0.1 / (2 ** i)
    dummy1, dummy2, dummy3, error = solve_BVP(dt)
    A5[i] = error

# Problem 2

x_i = -1
x_f = 1
t_f = 1
dt = 0.01
dx = 0.01
t = np.arange(0, t_f + dt, dt)
x = np.arange(-1, x_f + dx, dx)
len_x = len(x)
len_t = len(t)
mu = dt / (2 * (dx ** 2))
main_diag = (1 + 2 * mu) * np.ones(len_x - 2)
second_diag = -mu * np.ones(len_x - 3)
A = np.diag(main_diag) + np.diag(second_diag, 1) + np.diag(second_diag, -1)
main_diag = (1 - 2 * mu) * np.ones(len_x - 2)
second_diag = mu * np.ones(len_x - 3)
B = np.diag(main_diag) + np.diag(second_diag, 1) + np.diag(second_diag, -1)

A7 = A
A8 = B
A9 = 0
A10 = 0
A11 = 0

u = np.zeros(len_x)
u_sol = np.zeros((len_t, len_x))
u_a = 0
u_b = 0
for i in range (1, len(u) - 1):
    u[i] = np.exp(1 - (1 / (1 - x[i] ** 2)))
for i in range(len(t)):
    b = B @ u[1 : -1]
    b[0] += mu * u_a
    b[-1] += mu * u_b
    if (i == 0): A9 = b
    u[1 : -1] = np.linalg.solve(A, b)
    u_sol[i, :] = u

A10 = u_sol[49, :]
A11 = u_sol[-2, :]

ax = plt.axes(projection='3d')
X, Y = np.meshgrid(t, x)
Z = u_sol.T
ax.plot_surface(X, Y, Z, cmap = cm.coolwarm)
plt.axis([0, 4, 0, 1])
ax.view_init(30, 60)
plt.show()

A12 = X
A13 = Y
A14 = Z