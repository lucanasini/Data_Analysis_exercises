from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_openml
import logging
from torch import nn, optim
import torch
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np

logging.basicConfig(
        level  = logging.INFO,
        format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
logger = logging.getLogger("EX_2.2")

SEED = 42

np.random.seed(SEED)
torch.manual_seed(SEED)

class LinearModel(nn.Module):
    def __init__(self, n_inputs, n_ouputs):
        super().__init__()
        self.linear = nn.Linear(n_inputs, n_ouputs)

    def forward(self, x):
        return self.linear(x)

ames = fetch_openml(name="house_prices", as_frame=True)

X = ames.data
y = ames.target

logger.debug(X.shape)
logger.debug(y.shape)
logger.debug(X.head())
logger.debug(y.head())

X = X.select_dtypes(include=["number"]).fillna(0).values
y = y.astype(float).values

train_frac = np.linspace(0.1, 0.9, 20)

TRAIN = []
TEST = []

for i, frac in enumerate(train_frac):
    logger.info(f"Try: {i}")
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=frac, shuffle=True, random_state=SEED)

    scaler_X = StandardScaler()
    scaler_y = StandardScaler()

    X_train = scaler_X.fit_transform(X_train)
    X_test = scaler_X.transform(X_test)
    y_train = scaler_y.fit_transform(y_train.reshape(-1, 1))
    y_test = scaler_y.transform(y_test.reshape(-1, 1))

    X_train = torch.tensor(X_train, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32)

    model = LinearModel(X_train.shape[1], 1)

    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    TRAIN_LOSS = []
    TEST_LOSS = []
    for epoch in range(1000):
        model.train()
        # forward
        y_pred = model(X_train)
        train_loss = criterion(y_pred, y_train)

        # backward
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()

        TRAIN_LOSS.append(train_loss.detach().numpy())

        model.eval()
        with torch.no_grad():
            y_pred_test = model(X_test)
            test_loss = criterion(y_pred_test, y_test)
        TEST_LOSS.append(test_loss.numpy())

        if epoch % 100 == 0:
            logger.info(f"Epoch {epoch}, Train Loss: {train_loss.item()} | Test Loss: {test_loss.item()}")
    
    TRAIN.append(TRAIN_LOSS[-1])
    TEST.append(TEST_LOSS[-1])

    # plt.figure()
    # plt.plot(TRAIN_LOSS)
    # plt.plot(TEST_LOSS)
    # plt.yscale('log')
    # plt.show()

plt.figure()
plt.plot(train_frac, TRAIN, label='train', c='r', ls='-', marker='.')
plt.plot(train_frac, TEST, label='test', c='b', ls='--', marker='.')
plt.xlabel("Train Fraction")
plt.ylabel("Loss (MSE)")
plt.legend()
plt.tight_layout()
plt.savefig("loss_vs_train_fraction.pdf")
plt.show()