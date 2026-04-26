import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error


SEED = 42
np.random.seed(SEED)
data_degree = 2
max_degree_try = 50


def polynomial_func(X, coeffs):
    f = 0
    for i, c in enumerate(coeffs):
        f += c * X**i
    return f

def generate_data(X, coeffs, noise_mean, noise_std):
    f = polynomial_func(X, coeffs)
    n = len(X)
    scale = np.std(f)
    noise = np.random.normal(noise_mean, noise_std * scale, n)
    return f + noise



err_params = [0., 0.1]
func_params = np.random.uniform(0.1, 10, data_degree)

X_train = np.random.uniform(-1, 1, size=1000)
y_train = generate_data(X_train, func_params, *err_params)
X_train = X_train.reshape(-1, 1)

X_test = np.random.uniform(-1, 1, size=500)
y_test = generate_data(X_test, func_params, *err_params)
X_test = X_test.reshape(-1, 1)

X_grid = np.linspace(-1, 1, 500).reshape(-1, 1)


# polynomial regression
plt.figure(figsize=(10, 6))
plt.scatter(X_train, y_train, c="r", marker=".", label="data")

degs = range(1, max_degree_try+1)
err_train_poly = []
err_test_poly = []
for deg in degs:
    poly = make_pipeline(
        PolynomialFeatures(degree=deg),
        LinearRegression()
    )
    poly.fit(X_train, y_train)
    y_pred = poly.predict(X_train)
    err_train_poly.append(mean_squared_error(y_train, y_pred))
    y_test_pred = poly.predict(X_test)
    err_test_poly.append(mean_squared_error(y_test, y_test_pred))

    if deg % 10 == 1:
        y_grid_pred = poly.predict(X_grid)
        plt.plot(X_grid, y_grid_pred, label=f"degree={deg}")

plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.title("polynomial regression")
plt.tight_layout()
plt.show()


# knn
plt.figure(figsize=(10, 6))
plt.scatter(X_train, y_train, c="r", marker=".", label="data")

neighbors = range(1, 51)
err_train_neigh = []
err_test_neigh = []
for n in neighbors:
    neigh = KNeighborsRegressor(n_neighbors=n)
    neigh.fit(X_train, y_train)
    y_pred = neigh.predict(X_train)
    err_train_neigh.append(mean_squared_error(y_train, y_pred))
    y_test_pred = neigh.predict(X_test)
    err_test_neigh.append(mean_squared_error(y_test, y_test_pred))

    if n % 10 == 1:
        y_grid_pred = neigh.predict(X_grid)
        plt.plot(X_grid, y_grid_pred, label=f"k={n}")

plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.title("k-NN")
plt.tight_layout()
plt.show()


# empirical error plots
plt.figure(figsize=(10, 6))
plt.plot(degs, err_train_poly, label="polynomial regression")
plt.plot(neighbors, err_train_neigh, label="knn")
plt.xlabel("degree / neighbors")
plt.ylabel("empirical error")
plt.legend()
plt.title("empirical training error")
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(degs, err_test_poly, label="polynomial regression")
plt.plot(neighbors, err_test_neigh, label="knn")
plt.xlabel("degree / neighbors")
plt.ylabel("empirical error")
plt.legend()
plt.title("empirical test error")
plt.tight_layout()
plt.show()











polynomial_degrees = range(2, 101, 10)
for degree in polynomial_degrees:

    err_params = [0., 0.1]
    func_params = np.random.uniform(0.1, 10, degree)

    X_train = np.random.uniform(-10, 10, size=1000)
    y_train = generate_data(X_train, func_params, *err_params)
    X_train = X_train.reshape(-1, 1)

    X_test = np.random.uniform(-10, 10, size=500)
    y_test = generate_data(X_test, func_params, *err_params)
    X_test = X_test.reshape(-1, 1)

    X_grid = np.linspace(-10, 10, 500).reshape(-1, 1)


    # polynomial regression
    plt.figure(figsize=(10, 6))
    plt.scatter(X_train, y_train, c="r", marker=".", label="data")

    degs = range(1, 100, 1)
    err_train_poly = []
    err_test_poly = []
    for deg in degs:
        poly = make_pipeline(
            PolynomialFeatures(degree=deg),
            LinearRegression()
        )
        poly.fit(X_train, y_train)
        y_pred = poly.predict(X_train)
        err_train_poly.append(mean_squared_error(y_train, y_pred))
        y_test_pred = poly.predict(X_test)
        err_test_poly.append(mean_squared_error(y_test, y_test_pred))

        if deg % 10 == 1:
            y_grid_pred = poly.predict(X_grid)
            plt.plot(X_grid, y_grid_pred, label=f"degree={deg}")

    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.title("polynomial regression")
    plt.tight_layout()
    plt.show()


    # knn
    plt.figure(figsize=(10, 6))
    plt.scatter(X_train, y_train, c="r", marker=".", label="data")

    neighbors = range(1, 100, 1)
    err_train_neigh = []
    err_test_neigh = []
    for n in neighbors:
        neigh = KNeighborsRegressor(n_neighbors=n)
        neigh.fit(X_train, y_train)
        y_pred = neigh.predict(X_train)
        err_train_neigh.append(mean_squared_error(y_train, y_pred))
        y_test_pred = neigh.predict(X_test)
        err_test_neigh.append(mean_squared_error(y_test, y_test_pred))

        if n % 10 == 1:
            y_grid_pred = neigh.predict(X_grid)
            plt.plot(X_grid, y_grid_pred, label=f"k={n}")

    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.title("k-NN")
    plt.tight_layout()
    plt.show()


    # empirical error plots
    plt.figure(figsize=(10, 6))
    plt.plot(degs, err_train_poly, label="polynomial regression")
    plt.plot(neighbors, err_train_neigh, label="knn")
    plt.xlabel("degree / neighbors")
    plt.ylabel("empirical error")
    plt.legend()
    plt.title("empirical training error")
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.plot(degs, err_test_poly, label="polynomial regression")
    plt.plot(neighbors, err_test_neigh, label="knn")
    plt.xlabel("degree / neighbors")
    plt.ylabel("empirical error")
    plt.legend()
    plt.title("empirical test error")
    plt.tight_layout()
    plt.show()