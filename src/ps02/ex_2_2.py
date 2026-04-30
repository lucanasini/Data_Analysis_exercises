from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_openml
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import logging
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

os.makedirs("results", exist_ok=True)
filename = os.path.splitext(os.path.basename(__file__))[0]

logging.basicConfig(
        level  = logging.INFO,
        format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
logger = logging.getLogger("EX_2_2")


SEED = 42
np.random.seed(SEED)
test_splits = np.linspace(0.1, 0.9, 20)
shuffle = True


ames = fetch_openml(name="house_prices", as_frame=True)
X = ames.data
y = ames.target

logger.debug(X.shape)
logger.debug(y.shape)
logger.debug(X.head())
logger.debug(y.head())
logger.debug(X.dtypes.to_string())

# one-hot-encoding
X_num = X.select_dtypes(include=["int64", "float64"])
X_num = X_num.fillna(X_num.median())
X_cat = X.select_dtypes(include=["str"])
X_cat = X_cat.fillna(X_cat.mode().iloc[0])
X_cat_encoded = pd.get_dummies(X_cat, drop_first=True)
X_all = pd.concat([X_num, X_cat_encoded], axis=1).values
X_num = X_num.values
X_cat_encoded = X_cat_encoded.values

y = y.astype(float).values


train_loss_all = []
test_loss_all  = []
train_loss_num = []
test_loss_num  = []
train_loss_cat = []
test_loss_cat  = []

for i, frac in enumerate(test_splits):

    logger.info(f"Split: {i+1}/{len(test_splits)}")

    X_all_train, X_all_test, y_all_train, y_all_test = train_test_split(X_all, y, test_size=frac, shuffle=shuffle, random_state=SEED)

    scaler_X_all = StandardScaler()
    scaler_y_all = StandardScaler()
    X_all_train = scaler_X_all.fit_transform(X_all_train)
    X_all_test  = scaler_X_all.transform(X_all_test)
    y_all_train = scaler_y_all.fit_transform(y_all_train.reshape(-1, 1))
    y_all_test  = scaler_y_all.transform(y_all_test.reshape(-1, 1))
    
    model_all = LinearRegression()
    model_all.fit(X_all_train, y_all_train)
    y_all_pred = model_all.predict(X_all_train)
    train_loss_all.append(mean_squared_error(y_all_train, y_all_pred))
    y_all_test_pred = model_all.predict(X_all_test)
    test_loss_all.append(mean_squared_error(y_all_test, y_all_test_pred))


    X_num_train, X_num_test, y_num_train, y_num_test = train_test_split(X_num, y, test_size=frac, shuffle=shuffle, random_state=SEED)

    scaler_X_num = StandardScaler()
    scaler_y_num = StandardScaler()
    X_num_train = scaler_X_num.fit_transform(X_num_train)
    X_num_test  = scaler_X_num.transform(X_num_test)
    y_num_train = scaler_y_num.fit_transform(y_num_train.reshape(-1, 1))
    y_num_test  = scaler_y_num.transform(y_num_test.reshape(-1, 1))
    
    model_num = LinearRegression()
    model_num.fit(X_num_train, y_num_train)
    y_num_pred = model_num.predict(X_num_train)
    train_loss_num.append(mean_squared_error(y_num_train, y_num_pred))
    y_num_test_pred = model_num.predict(X_num_test)
    test_loss_num.append(mean_squared_error(y_num_test, y_num_test_pred))


    X_cat_train, X_cat_test, y_cat_train, y_cat_test = train_test_split(X_cat_encoded, y, test_size=frac, shuffle=shuffle, random_state=SEED)

    scaler_X_cat = StandardScaler()
    scaler_y_cat = StandardScaler()
    X_cat_train = scaler_X_cat.fit_transform(X_cat_train)
    X_cat_test  = scaler_X_cat.transform(X_cat_test)
    y_cat_train = scaler_y_cat.fit_transform(y_cat_train.reshape(-1, 1))
    y_cat_test  = scaler_y_cat.transform(y_cat_test.reshape(-1, 1))
    
    model_cat = LinearRegression()
    model_cat.fit(X_cat_train, y_cat_train)
    y_cat_pred = model_cat.predict(X_cat_train)
    train_loss_cat.append(mean_squared_error(y_cat_train, y_cat_pred))
    y_cat_test_pred = model_cat.predict(X_cat_test)
    test_loss_cat.append(mean_squared_error(y_cat_test, y_cat_test_pred))


fig, ax = plt.subplots(1, 3, figsize=(15, 5))

ax[0].plot(test_splits, train_loss_all, label='train', c='r', ls='-')
ax[0].plot(test_splits, test_loss_all,  label='test',  c='b', ls='--')
ax[0].set_xlabel("Test Fraction")
ax[0].set_ylabel("Loss (MSE)")
ax[0].set_title("Numerical + Categorical Features")
ax[0].legend()

ax[1].plot(test_splits, train_loss_num, label='train', c='r', ls='-')
ax[1].plot(test_splits, test_loss_num,  label='test',  c='b', ls='--')
ax[1].set_xlabel("Test Fraction")
ax[1].set_title("Only Numerical Features")
ax[1].legend()

ax[2].plot(test_splits, train_loss_cat, label='train', c='r', ls='-')
ax[2].plot(test_splits, test_loss_cat,  label='test',  c='b', ls='--')
ax[2].set_xlabel("Test Fraction")
ax[2].set_title("Only Categorical Features")
ax[2].legend()

plt.tight_layout()
plt.savefig(f"results/{filename}_loss_vs_train_fraction.pdf")
plt.show()
