'''
Empirical Rademacher Complexity     C(H) = E(sup 1/n sum(sigma*h))
'''
from sklearn import datasets
import numpy as np
import os
os.makedirs("results", exist_ok=True)

iris = datasets.load_iris()
X = iris.data[:, 1]

target_mask = iris.target != 2
X = X[target_mask]

def decision_stumps(x, a):
    return 2 * (x > a) - 1

def emp_rad_compl(X, func, n_trial):
    n = len(X)
    sorted_idx = np.argsort(X)
    X_sorted = X[sorted_idx]

    estimates = []
    for _ in range(n_trial):
        sigma = np.random.choice([-1, 1], size=n)

        best = -np.inf

        for i in range(n):
            a = X_sorted[i]
            h = func(X, a)

            score = 1/n * np.sum(sigma * h)
            best = max(best, score)

        estimates.append(best)

    mean = np.mean(estimates)
    return mean

num_iterations = 5000
print(emp_rad_compl(X, decision_stumps, num_iterations))
with open(f"results/ex_1_12.txt", "a") as f:
        f.write(f"Total iterations: {num_iterations:4}\t-\tE.R.C.:\t{emp_rad_compl(X, decision_stumps, num_iterations):.5f}\n")
