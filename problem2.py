import numpy as np
import csv
from matplotlib import pyplot as plt
from functools import partial
from scipy.optimize import least_squares

# Hyperparameters
T_max = 119

# Load data
reader = csv.reader(open('project10_data.csv', 'r'))
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
    I[t] = accumulated_cases[t + t0 + 7] - accumulated_cases[t + t0 -7]

plt.figure()
plt.plot(I)
plt.savefig("I.png")

# Question 2: implement an Euler schema for the SIR model

def SIR_simulation(S0, I0, R0, beta, alpha, T0, T_max):
    transition_matrix = lambda p: np.array(
        [
            [np.exp(-beta * T0 * I0), 0, 0],
            [
                beta * I0 / (alpha - beta * I0) * np.exp(-beta * T0 * I0) - 
                beta * I0 / (alpha - beta * I0) * np.exp(-alpha * T0),
                np.exp(-alpha * T0),
                0
            ],
            [
                1 - alpha / (alpha - beta * I0) * np.exp(-beta * T0 * I0) +
                beta * I0 / (alpha - beta * I0) * np.exp(-alpha * T0),
                1 - np.exp(-alpha * T0),
                1
            ]
        ]
    )

    def euler_step(S, I, R, transition_matrix):
        return np.matmul(transition_matrix, np.array([S, I, R]))

    results = [
        [S0, I0, R0]
    ]

    for t in range(1, T_max + 1):
        S0, I0, R0 = euler_step(S0, I0, R0, transition_matrix(t))
        results.append([S0, I0, R0])

    S = np.array([result[0] for result in results])
    I = np.array([result[1] for result in results])
    R = np.array([result[2] for result in results])

    return S, I, R

def SIR_func(x):
    # h = 0.01
    R0 = 0
    I0 = I[0]
    T0 = 0.01

    S0 = x[0]
    beta = x[1]
    alpha = x[2]
    
    _, I_hat, _ = SIR_simulation(S0, I0, R0, beta, alpha, T0, T_max)

    return I - I_hat

# Least squares
minimum_population = 0 #accumulated_cases[T_max + t0]
x0 = np.array([minimum_population, 1, 1])
bounds = ([minimum_population, 0, 0.2], [population, 10, 1])
res = least_squares(SIR_func, x0, bounds=bounds)
print(res)
S, I_hat, R = SIR_simulation(res.x[0], I[0], 0, res.x[1], res.x[2], 0.01, T_max)
print(f"Least squares: S0 = {res.x[0]}, beta = {res.x[1]}, alpha = {res.x[2]}")

plt.figure()
plt.plot(S, label="Susceptible")
plt.plot(I, label="Infected")
plt.plot(I_hat, label="Infected (estimated)")
plt.plot(R, label="Recovered")
plt.legend()
plt.savefig("SIR.png")

# Question 3: omega simulation
omegas = []

for alpha in [
    1/10, 1/9, 1/8, 1/7, 1/6, 1/5
]:
    for R0 in [
        0.8, 0.9, 0.95, 1, 1.05, 1.1, 1.15, 1.2, 1.3, 1.4, 1.5, 1.6
    ]:
        for N in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
            N_max = population * N

            omegas.append((alpha, R0 * alpha, N_max))

print("Question 3: omega simulation begins")
Js = []
for alpha, beta, N_max in omegas:
    S, I_hat, R = SIR_simulation(N_max, I[0], 0, beta, alpha, 0.01, T_max)
    J = np.sum((I - I_hat) ** 2)
    Js.append(J)

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
