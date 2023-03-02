import numpy as np
import csv
from matplotlib import pyplot as plt
from functools import partial
from scipy.optimize import least_squares
from scipy.integrate import odeint

# Hyperparameters
T_max = 119

# Load data
reader = csv.reader(open("project10_data.csv", "r"))
titles = np.array(next(reader))
data = np.array(list(reader))

dates = titles[12:]
population = data[0, 11].astype(dtype=int)
accumulated_cases = data[0, 12:].astype(dtype=int)
accumulated_deaths = data[1, 12:].astype(dtype=int)

t0 = np.argwhere(accumulated_cases > 5)[1, 0]
print(f"Simulation starts on {t0}, which is {dates[t0]}")

# Question 1: calculate the rate of active infection
T_max = 119
I = np.zeros(T_max + 1)

for t in range(0, T_max + 1):
    I[t] = accumulated_cases[t + t0 + 7] - accumulated_cases[t + t0 - 7]

plt.figure()
plt.plot(I)
plt.savefig("I.png")

# Question 2: implement an Euler schema for the SIR model


def f(y, t, alpha, beta, N):
    S, I, R = y
    d0 = -beta * S * I/N  # derivative of S(t)
    d1 = beta * S * I/N - alpha * I  # derivative of I(t)
    d2 = alpha * I  # derivative of R(t)
    return [d0, d1, d2]


def SIR_simulation(x, return_all=False):
    alpha, beta, N = x
    y_0 = [N, I[0], 0]  # Susceptible, Infected, Recovered

    t = np.arange(start=1, stop=T_max+1.01, step=0.01)
    y = odeint(partial(f, alpha=alpha, beta=beta, N=N), y_0, t)
    y = y[::100]

    if return_all:
        return y[:, 0], y[:, 1], y[:, 2]

    return I - y[:, 1]


# Question 3: omega simulation
omegas = []

for alpha in [1 / 10, 1 / 9, 1 / 8, 1 / 7, 1 / 6, 1 / 5]:
    for R0 in [0.8, 0.9, 0.95, 1, 1.05, 1.1, 1.15, 1.2, 1.3, 1.4, 1.5, 1.6]:
        for N in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
            N_max = population * N
            omegas.append((alpha, R0 * alpha, N_max))

print("Question 3: omega simulation begins")
Js = []
for alpha, beta, N_max in omegas:
    S, I_hat, R = SIR_simulation((alpha, beta, N_max), return_all=True)
    J = np.sqrt(np.sum((I - I_hat) ** 2))
    Js.append(J)

# print(Js)
omegas = np.array(omegas)
Js = np.array(Js)
print("Question 3: omega simulation ends")

# Question 4: plot the results
plt.figure()
plt.scatter(omegas[:, 1], omegas[:, 2], c=Js)

# Also plot the minimum
min_J = np.argmin(Js)
plt.scatter(omegas[min_J, 1], omegas[min_J, 2], c="red", marker="x", label="Minimum")

plt.xlabel("beta")
plt.ylabel("N")
plt.title("J vs beta and N")
plt.colorbar()
plt.legend()
plt.savefig("J_vs_beta_and_N.png")


# Visualize the best fit
alpha, beta, N_max = omegas[min_J]
S, I_hat, R = SIR_simulation((alpha, beta, N_max), return_all=True)
plt.figure()
plt.plot(I, label="I")
# print(S * N_max, I_hat * N_max)
# plt.plot(S * N_max, label="S")
plt.plot(I_hat, label="I_hat")
# plt.plot(R * N_max, label="R")

plt.legend()
plt.savefig("I_vs_I_hat.png")
