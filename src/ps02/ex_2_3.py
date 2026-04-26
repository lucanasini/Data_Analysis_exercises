import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler


SEED = 42
np.random.seed(SEED)
max_patience = 5
relative_eps = 1e-3
data_degree = 10
max_degree_try = 50


def polynomial_func(X, coeffs):
    f = 0
    for i, c in enumerate(coeffs):
        f += c * X**i
    return f

def generate_data_uniform(X, coeffs, noise_mean, noise_std):
    f = polynomial_func(X, coeffs)
    n = len(X)
    noise = np.random.normal(noise_mean, noise_std, n)
    return f + noise

def generate_data_non_uniform(X, coeffs, noise_mean, noise_std):
    f = polynomial_func(X, coeffs)
    n = len(X)
    sigma = noise_std * (1 + X**2)
    noise = np.random.normal(noise_mean, sigma, n)
    return f + noise


err_params = [0., 0.1]
func_params = np.random.uniform(1., 5., data_degree)

X_train = np.random.uniform(-1., 1., size=1000)
y_train_nu = generate_data_non_uniform(X_train, func_params, *err_params)
y_train_u = generate_data_uniform(X_train, func_params, *err_params)
X_train = X_train.reshape(-1, 1)

X_test = np.random.uniform(-1., 1., size=500)
y_test_nu = generate_data_non_uniform(X_test, func_params, *err_params)
y_test_u = generate_data_uniform(X_test, func_params, *err_params)
X_test = X_test.reshape(-1, 1)

X_grid = np.linspace(-1., 1., 500).reshape(-1, 1)


plt.figure(figsize=(10, 6))
plt.scatter(X_train, y_train_u, c="g", marker=".", label="data uniform")
plt.scatter(X_train, y_train_nu, c="r", marker=".", label="data non uniform")


degs = range(1, max_degree_try+1)
loss_train_u = []
loss_test_u = []
best_loss_u = float(np.inf)
current_patience_u = max_patience
degree_list_u = []
loss_train_nu = []
loss_test_nu = []
best_loss_nu = float(np.inf)
current_patience_nu = max_patience
degree_list_nu = []

for deg in degs:

    if current_patience_u == 0:
        break
    
    poly_u = make_pipeline(
        PolynomialFeatures(degree=deg),
        LinearRegression()
    )
    poly_u.fit(X_train, y_train_u)
    y_pred_u = poly_u.predict(X_train)
    loss_train_u.append(mean_squared_error(y_train_u, y_pred_u))
    y_test_pred_u = poly_u.predict(X_test)
    loss_u = mean_squared_error(y_test_u, y_test_pred_u)
    loss_test_u.append(loss_u)
    
    if loss_u < best_loss_u * (1 - relative_eps):
        best_loss_u = loss_u
        current_patience_u = max_patience
    else:
        current_patience_u -= 1
    
    degree_list_u.append(deg)
    

for deg in degs:

    if current_patience_nu == 0:
        break

    poly_nu = make_pipeline(
        PolynomialFeatures(degree=deg),
        LinearRegression()
    )
    poly_nu.fit(X_train, y_train_nu)
    y_pred_nu = poly_nu.predict(X_train)
    loss_train_nu.append(mean_squared_error(y_train_nu, y_pred_nu))
    y_test_pred_nu = poly_nu.predict(X_test)
    loss_nu = mean_squared_error(y_test_nu, y_test_pred_nu)
    loss_test_nu.append(loss_nu)

    if loss_nu < best_loss_nu * (1 - relative_eps):
        best_loss_nu = loss_nu
        current_patience_nu = max_patience
    else:
        current_patience_nu -= 1
    
    degree_list_nu.append(deg)


plt.figure(figsize=(10, 6))
plt.plot(degree_list_u, loss_train_u, c='r', ls='-', lw=1, label="train uniform")
plt.plot(degree_list_u, loss_test_u, c='b', ls='--', lw=1, label="test uniform")
plt.plot(degree_list_nu, loss_train_nu, c='r', ls='-', lw=3, label="train non uniform")
plt.plot(degree_list_nu, loss_test_nu, c='b', ls='--', lw=3, label="test non uniform")
plt.xlabel("degree")
plt.ylabel("loss")
plt.yscale("log")
plt.legend()
plt.tight_layout()
plt.show()

print(f"Best Loss (uniform):     {best_loss_u:.4}\n"
      f"Best Loss (non uniform): {best_loss_nu:.4}")