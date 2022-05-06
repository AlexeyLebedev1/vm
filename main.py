import time

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Button, TextBox

plt.rcParams.update({'figure.figsize': '7.5, 6', "figure.facecolor": 'lightblue', 'axes.edgecolor': 'black'})


def solver(I, V, f, c, L, dt, C, T, user_action=None):
    """Solve u_tt=c^2*u_xx + f on (0,L)x(0,T]."""
    Nt = int(round(T / dt))
    t = np.linspace(0, Nt * dt, Nt + 1)  # Mesh points in time
    dx = dt * c / float(C)
    Nx = int(round(L / dx))
    x = np.linspace(0, L, Nx + 1)  # Mesh points in space
    C2 = C ** 2  # Help variable in the scheme
    # Make sure dx and dt are compatible with x and t
    dt = t[1] - t[0]

    if f is None or f == 0:
        f = lambda x, t: 0
    if V is None or V == 0:
        V = lambda x: 0

    u = np.zeros(Nx + 1)  # Solution array at new time level
    u_n = np.zeros(Nx + 1)  # Solution at 1 time level back
    u_nm1 = np.zeros(Nx + 1)  # Solution at 2 time levels back

    # Load initial condition into u_n
    for i in range(0, Nx + 1):
        u_n[i] = I(x[i])

    if user_action is not None:
        user_action(u_n, x, t, 0)

    # Special formula for first time step
    n = 0
    for i in range(1, Nx):
        u[i] = u_n[i] + dt * V(x[i]) + \
               0.5 * C2 * (u_n[i - 1] - 2 * u_n[i] + u_n[i + 1]) + \
               0.5 * dt ** 2 * f(x[i], t[n])
    u[0] = 0
    u[Nx] = 0

    if user_action is not None:
        user_action(u, x, t, 1)

    # Switch variables before next step
    u_nm1[:] = u_n
    u_n[:] = u

    cr = 0

    for n in range(1, Nt):
        # Update all inner points at time t[n+1]
        for i in range(1, Nx):
            u[i] = - u_nm1[i] + 2 * u_n[i] + \
                   C2 * (u_n[i - 1] - 2 * u_n[i] + u_n[i + 1]) + \
                   dt ** 2 * f(x[i], t[n]) 

        # Insert boundary conditions
        u[0] = 0
        u[Nx] = 0
        if user_action is not None:
            if user_action(u, x, t, n + 1):
                break

        # Switch variables before next step
        u_nm1[:] = u_n
        u_n[:] = u

        graph_axes.clear()
        graph_axes.grid()
        graph_axes.plot(x, u, 'r')
        plt.pause(0.05)


fig, graph_axes = plt.subplots()
graph_axes.grid()
fig.subplots_adjust(left=0.12, right=0.93, top=0.98, bottom=0.4)
graph_axes.set_xlabel('X')
graph_axes.set_ylabel('T')


def set_value(var, text):
    try:
        globals().__setitem__(var, float(text))
    except ValueError:
        pass


# User parameters

l = 0.80
lbox = TextBox(plt.axes([0.15, 0.25, 0.10, 0.07]), 'l = ', initial=l)
lbox.on_submit(lambda text: (
    set_value('l', text),
    graph_axes.set_xlim(l)))

T = 0.1
Tbox = TextBox(plt.axes([0.15, 0.15, 0.10, 0.07]), 'T = ', initial=T)
Tbox.on_submit(lambda text: set_value('T', text))

dx = 0.001
dxbox = TextBox(plt.axes([0.38, 0.15, 0.10, 0.07]), 'dx = ', initial=dx)
dxbox.on_submit(lambda text: set_value('dx', text))

dt = 0.001
dtbox = TextBox(plt.axes([0.38, 0.25, 0.10, 0.07]), 'dt = ', initial=dt)
dtbox.on_submit(lambda text: set_value('dt', text))

a = 2
abox = TextBox(plt.axes([0.56, 0.25, 0.10, 0.07]), 'a = ', initial=a)
abox.on_submit(lambda text: set_value('a', text))

b = 0.005
bbox = TextBox(plt.axes([0.56, 0.15, 0.10, 0.07]), 'b = ', initial=b)
bbox.on_submit(lambda text: set_value('b', text))

x0 = round(0.8 * l, 2)
x0box = TextBox(plt.axes([0.74, 0.25, 0.10, 0.07]), 'x0 = ', initial=x0)
x0box.on_submit(lambda text: set_value('x0', text))


def solve_and_draw(event):
    def I(x):
        return b * x / x0 if x < x0 else b / (l - x0) * (l - x)

    def viz(u, x, t, n):
        #TODO: plot u(x) in time t[n-1]
        pass

    solver(I, 0, 0, a, l, dt, 0.85, T, viz)


solve_btn = Button(plt.axes([0.37, 0.05, 0.28, 0.075]), 'solve')
solve_btn.on_clicked(solve_and_draw)

clear_btn = Button(plt.axes([0.07, 0.05, 0.28, 0.075]), 'clear')
clear_btn.on_clicked(lambda event: (
    graph_axes.clear(),
    graph_axes.grid(),
    plt.draw()))

plt.show()
