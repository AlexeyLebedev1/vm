import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Button, TextBox
from scipy import  integrate
from copy import copy

plt.rcParams.update({'figure.figsize': '9, 7', "figure.facecolor": 'lightblue', 'axes.edgecolor': 'black'})


def solver(I, V, f, a, L, dt, dx, T, user_action=None):
    """Solve u_tt=a^2*u_xx + f on (0,L)x(0,T]."""
    Nt = int(round(T / dt))
    t = np.linspace(0, Nt * dt, Nt + 1)  # Mesh points in time
    Nx = int(round(L / dx))
    x = np.linspace(0, L, Nx + 1)  # Mesh points in space
    C2 = (dt * a / dx) ** 2  # Help variable in the scheme
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

    left, right = u_n[0], u_n[Nx]
    u[0] = left
    u[Nx] = right

    if user_action is not None:
        user_action(u, x, t, 1)

    # Switch variables before next step
    u_nm1[:] = u_n
    u_n[:] = u

    for n in range(1, Nt):
        # Update all inner points at time t[n+1]
        for i in range(1, Nx):
            u[i] = - u_nm1[i] + 2 * u_n[i] + \
                   C2 * (u_n[i - 1] - 2 * u_n[i] + u_n[i + 1]) + \
                   dt ** 2 * f(x[i], t[n])

        # Insert boundary conditions
        u[0] = left
        u[Nx] = right
        if user_action is not None:
            if user_action(u, x, t, n + 1):
                break

        # Switch variables before next step
        u_nm1[:] = u_n
        u_n[:] = u

    return u, x, t


fig, graph_axes = plt.subplots()
graph_axes.grid()
fig.subplots_adjust(left=0.12, right=0.93, top=0.98, bottom=0.4)
graph_axes.set_xlabel('X')
graph_axes.set_ylabel('U')


def set_value(var, text):
    try:
        globals().__setitem__(var, float(text))
    except ValueError:
        pass


# User parameters

l = 0.75
lbox = TextBox(plt.axes([0.05, 0.25, 0.1, 0.07]), 'l = ', initial=l)
lbox.on_submit(lambda text: set_value('l', text))

T = 0.1
Tbox = TextBox(plt.axes([0.05, 0.15, 0.1, 0.07]), 'T = ', initial=T)
Tbox.on_submit(lambda text: set_value('T', text))

dx = 0.023
dxbox = TextBox(plt.axes([0.2, 0.15, 0.10, 0.07]), 'dx = ', initial=dx)
dxbox.on_submit(lambda text: set_value('dx', text))

dt = 0.001
dtbox = TextBox(plt.axes([0.2, 0.25, 0.10, 0.07]), 'dt = ', initial=dt)
dtbox.on_submit(lambda text: set_value('dt', text))

a = 20
abox = TextBox(plt.axes([0.35, 0.25, 0.10, 0.07]), 'a = ', initial=a)
abox.on_submit(lambda text: set_value('a', text))


fi1 = 0.3
fi1box = TextBox(plt.axes([0.65, 0.25, 0.10, 0.07]), 'fi:', label_pad = 0.1, initial=fi1)
fi1box.on_submit(lambda text: set_value('fi1', text))

fi2 = 0.3
fi2box = TextBox(plt.axes([0.75, 0.25, 0.10, 0.07]), '', initial=fi2)
fi2box.on_submit(lambda text: set_value('fi2', text))

fi3 = 0.3
fi3box = TextBox(plt.axes([0.85, 0.25, 0.10, 0.07]), '', initial=fi3)
fi3box.on_submit(lambda text: set_value('fi3', text))

psi1 = 0.1
psi1box = TextBox(plt.axes([0.65, 0.15, 0.10, 0.07]), 'psi:', label_pad = 0.1, initial=psi1)
psi1box.on_submit(lambda text: set_value('psi1', text))

psi2 = 0.1
psi2box = TextBox(plt.axes([0.75, 0.15, 0.10, 0.07]), '', initial=psi2)
psi2box.on_submit(lambda text: set_value('psi2', text))

psi3 = 0.1
psi3box = TextBox(plt.axes([0.85, 0.15, 0.10, 0.07]), '', initial=psi3)
psi3box.on_submit(lambda text: set_value('psi3', text))

b1 = 0.1
b1box = TextBox(plt.axes([0.65, 0.05, 0.10, 0.07]), 'b:', label_pad=0.1, initial=b1)
b1box.on_submit(lambda text: set_value('b1', text))

b2 = 0.1
b2box = TextBox(plt.axes([0.75, 0.05, 0.10, 0.07]), '', initial=b2)
b2box.on_submit(lambda text: set_value('b2', text))

b3 = 0.1
b3box = TextBox(plt.axes([0.85, 0.05, 0.10, 0.07]), '', initial=b3)
b3box.on_submit(lambda text: set_value('b3', text))


def solve_and_draw(event):
    fi0 = np.sqrt((2. / l - (np.sqrt(fi1) + np.sqrt(fi2) + np.sqrt(fi3))) / 2.)
    psi0 = - (fi1 * psi1 + fi2 * psi2 + fi3 * psi3) / (2. * fi0)

    def I(x):
        return fi0 + fi1 * np.cos(np.pi * x / l) + fi2 * np.cos(2 * np.pi * x / l) + fi3 * np.cos(3 * np.pi * x / l)

    def V(x):
        return psi0 + psi1 * np.cos(np.pi * x / l) + psi2 * np.cos(2 * np.pi * x / l) + psi3 * np.cos(3 * np.pi * x / l)

    def F(u, x, t):

        def b():
            return b1 + b2 * np.cos(np.pi * x / l) + b3 * np.cos(2 * np.pi * x / l)

        def r():
            def f(x):
                return np.sqrt(u) - b() * np.sqrt(u)
            return integrate.quad(f, 0, l)
        return u * (b() + r())


    def viz(u, x, t, n):
        global I0
        graph_axes.clear()
        graph_axes.set_xlim([0, l])
        graph_axes.set_ylim([- (fi0 + 2), fi0 + 2])
        graph_axes.grid()
        if n == 0:
            I0 = copy(u)
        graph_axes.plot(x, u, 'r')
        graph_axes.plot(x, I0, 'b')
        graph_axes.legend(['t=%.3f'% t[n]], loc='lower left')
        plt.pause(0.00005)

    solver(I, V, 0, a, l, dt, dx, T, viz)


solve_btn = Button(plt.axes([0.30, 0.05, 0.20, 0.075]), 'solve')
solve_btn.on_clicked(solve_and_draw)

clear_btn = Button(plt.axes([0.05, 0.05, 0.20, 0.075]), 'clear')
clear_btn.on_clicked(lambda event: (
    graph_axes.clear(),
    graph_axes.grid(),
    plt.draw()))

plt.show()
