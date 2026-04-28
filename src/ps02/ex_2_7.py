from sklearn.model_selection import train_test_split, KFold, LeaveOneOut
from sklearn.datasets import fetch_openml
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
import logging
from sklearn.preprocessing import StandardScaler
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
shuffle = True
k_folds = [5, 7, 10]
cv_types = ["Hold Out"] + [f"{k}-Fold" for k in k_folds] + ["LOO"]


ames = fetch_openml(name="house_prices", as_frame=True)
X = ames.data
y = ames.target

# one-hot-encoding
X_num = X.select_dtypes(include=[np.number])
X_num = X_num.fillna(X_num.median())
X_cat = X.select_dtypes(include=["str", "category"])
X_cat = X_cat.fillna(X_cat.mode().iloc[0])
X_cat_encoded = pd.get_dummies(X_cat, drop_first=True)
X = pd.concat([X_num, X_cat_encoded], axis=1).values

y = y.astype(float).values


loss_train = {k: [] for k in cv_types}
loss_test  = {k: [] for k in cv_types}
best_loss  = {k: float("inf") for k in cv_types}


# holdout
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, shuffle=shuffle, random_state=SEED)

scaler_X = StandardScaler()
scaler_y = StandardScaler()
X_train = scaler_X.fit_transform(X_train)
X_test  = scaler_X.transform(X_test)
y_train = scaler_y.fit_transform(y_train.reshape(-1, 1))
y_test  = scaler_y.transform(y_test.reshape(-1, 1))

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_train)
train_loss = mean_squared_error(y_train, y_pred)
loss_train["Hold Out"].append(train_loss)
y_test_pred = model.predict(X_test)
loss_test["Hold Out"].append(mean_squared_error(y_test, y_test_pred))

logger.info("Hold Out done.")

# kfold
for k in k_folds:

    kfold = KFold(n_splits=k, shuffle=shuffle, random_state=SEED)

    train_losses = []
    test_losses  = []

    for fold, (train_idx, test_idx) in enumerate(kfold.split(X)):

        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        scaler_X = StandardScaler()
        scaler_y = StandardScaler()
        X_train = scaler_X.fit_transform(X_train)
        X_test  = scaler_X.transform(X_test)
        y_train = scaler_y.fit_transform(y_train.reshape(-1, 1))
        y_test  = scaler_y.transform(y_test.reshape(-1, 1))
        
        model = LinearRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_train)
        train_loss = mean_squared_error(y_train, y_pred)
        train_losses.append(train_loss)
        y_test_pred = model.predict(X_test)
        test_losses.append(mean_squared_error(y_test, y_test_pred))

    loss_train[f"{k}-Fold"].append(np.mean(train_losses))
    loss_test[f"{k}-Fold"].append(np.mean(test_losses))
    
    logger.info(f"{k}-Fold done.")


# loo
loo = LeaveOneOut()

train_losses = []
test_losses  = []

for fold, (train_idx, test_idx) in enumerate(loo.split(X)):

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    X_train = scaler_X.fit_transform(X_train)
    X_test  = scaler_X.transform(X_test)
    y_train = scaler_y.fit_transform(y_train.reshape(-1, 1))
    y_test  = scaler_y.transform(y_test.reshape(-1, 1))
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_train)
    train_loss = mean_squared_error(y_train, y_pred)
    train_losses.append(train_loss)
    y_test_pred = model.predict(X_test)
    test_losses.append(mean_squared_error(y_test, y_test_pred))

loss_train["LOO"].append(np.mean(train_losses))
loss_test["LOO"].append(np.mean(test_losses))

logger.info("LOO done.")


for i in cv_types:
    logger.info(f"{i}:\n"
                f"    Test Loss:  {loss_train[i][-1]:.4f}\n"
                f"    Train Loss: {loss_test[i][-1]:.4f}")