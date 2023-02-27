import numpy as np
import csv
from matplotlib import pyplot as plt
from functools import partial

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

# Question 1
t0 = np.argwhere(accumulated_cases > 5)[1, 0]
print(f"Simulation starts on {t0}, which is {dates[t0]}")
dates = dates[t0:]
accumulated_cases = accumulated_cases[t0:]

# Question 2
I_backup = accumulated_cases[:T_max+1]
N_max = population
N_min = I_backup[-1] + 1
print(f"N_min = {N_min}, N_max = {N_max}")

def jacobi(I, beta, N):
    J = 0
    for t in range(0, T_max+1):
        J += (beta * t - np.log(I[t] / (N - I[t])) + np.log(I[0] / (N - I[0]))) ** 2
    return J

## Question 2a Implement SI Algo 1 - Known N
for N in [N_min, N_max]:
    I = I_backup.copy() / N

    print(f"Running for N = {N}")

    summation = 0
    for t in range(1, T_max+1):
        summation += t * np.log((I[t] / I[0]) * (N - I[0]) / (N - I[t]))
    beta_hat = 6 / (T_max * (T_max + 1) * (2 * T_max + 1)) * summation
    print(f"beta_hat = {beta_hat}")

    # Objective function
    J = jacobi(I, beta_hat, N)
    print(f"J = {J}")

    prediction = N * I[0] / (I[0] + (N - I[0]) * np.exp(-beta_hat * np.arange(0, T_max+1)))
    plt.figure()
    plt.plot(prediction, label="Prediction")
    plt.plot(I, label="Actual")
    plt.legend()
    plt.savefig(f"SI_known_N (N={N}).png")
    print("-" * 20)

## Question 2b Implement SI Algo 2 - Unknown N

N = N_min
J_old = np.inf
a = 6 / (T_max * (T_max + 1) * (2 * T_max + 1))
all_J = []
considered_N = []

while True:
    I = I_backup.copy() / N

    summation_part_a = 0
    for t in range(1, T_max+1):
        summation_part_a += np.log((I[t] / I[0]) * (N - I[0]) / (N - I[t])) ** 2
    
    summation_part_b = 0
    for t in range(1, T_max+1):
        summation_part_b += t * np.log((I[t] / I[0]) * (N - I[0]) / (N - I[t]))
    
    J = summation_part_a - a * summation_part_b ** 2
    all_J.append(J)
    considered_N.append(N)

    if J > J_old:
        break

    J_old = J
    N += 1

beta_hat = a * summation_part_b
print(f"Unknown N: beta_hat = {beta_hat}, N = {N}, J = {J}")
print("-" * 20)

plt.figure()
plt.plot(all_J)
plt.savefig("SI_unknown_N_intermediate.png")

# Run for N = N_min to N = N_max
all_J = []

for N in range(N_min, N_max, 100):
    I = I_backup.copy() / N
    
    summation_part_a = 0
    for t in range(1, T_max+1):
        summation_part_a += np.log((I[t] / I[0]) * (N - I[0]) / (N - I[t])) ** 2
    
    summation_part_b = 0
    for t in range(1, T_max+1):
        summation_part_b += t * np.log((I[t] / I[0]) * (N - I[0]) / (N - I[t]))
    
    J = summation_part_a - a * summation_part_b ** 2
    all_J.append(J)

plt.figure()
plt.plot(all_J, label="all_J")
plt.plot(np.argmin(all_J), np.min(all_J), 'ro', label="min")
plt.legend()
plt.savefig("SI_unknown_N.png")

## Question 2c
for N in considered_N:
    print(f"N = {N}")
    I = I_backup.copy() / N

    summation = 0
    for t in range(1, T_max+1):
        summation += t * np.log((I[t] / I[0]) * (N - I[0]) / (N - I[t]))
    beta_hat = 6 / (T_max * (T_max + 1) * (2 * T_max + 1)) * summation
    print(f"beta_hat = {beta_hat}")

    # Compute Ideal Objective Function
    summation = 0
    for t in range(0, T_max+1):
        summation += (I[t] - N*I[0] / (I[0] + (N - I[0]) * np.exp(-beta_hat * t))) ** 2
    J = summation
    print(f"Ideal = {J}")
