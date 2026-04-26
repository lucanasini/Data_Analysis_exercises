from sklearn.datasets import load_digits
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression

SEED = 42
train_frac = 0.7

digits = load_digits()
X, y = digits.data, digits.target

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_frac, shuffle=True, random_state=SEED)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# knn
neighbors = range(1, 51)
loss_train_knn = []
loss_test_knn = []
for n in neighbors:
    knn = KNeighborsClassifier(n_neighbors=n)
    knn.fit(X_train, y_train)
    y_train_prob = knn.predict_proba(X_train)
    loss_train_knn.append(log_loss(y_train, y_train_prob))
    y_test_prob = knn.predict_proba(X_test)
    loss_test_knn.append(log_loss(y_test, y_test_prob))

best_test_loss_idx = np.argmin(loss_test_knn)
print(f"k - Nearest Neighbors:\n"
      f"    Train Loss: {loss_train_knn[best_test_loss_idx]:.4f} | "
      f"Test Loss: {loss_test_knn[best_test_loss_idx]:.4f} | "
      f"k: {neighbors[best_test_loss_idx]}")

plt.figure(figsize=(10, 6))
plt.plot(neighbors, loss_train_knn, c='r', ls='-', label="train")
plt.plot(neighbors, loss_test_knn, c='b', ls='--', label="test")
plt.xlabel("neighbors")
plt.ylabel("loss")
plt.legend()
plt.tight_layout()
plt.show()


# linear discriminant analysis
lda = LinearDiscriminantAnalysis()
lda.fit(X_train, y_train)
y_train_proba = lda.predict_proba(X_train)
loss_train_lda = log_loss(y_train, y_train_proba)
y_test_proba = lda.predict_proba(X_test)
loss_test_lda = log_loss(y_test, y_test_proba)

print(f"Linear Discriminant Analysis:\n"
      f"    Train Loss: {loss_train_lda:.4f} | "
      f"Test Loss: {loss_test_lda:.4f}")


# quadratic discriminant analysis
alpha = np.linspace(0, 0.9, 10)
loss_train_qda = []
loss_test_qda = []
alpha_vals = []
for a in alpha:
    try:
        qda = QuadraticDiscriminantAnalysis(reg_param=a)
        qda.fit(X_train, y_train)
        y_train_proba = qda.predict_proba(X_train)
        loss_train_qda.append(log_loss(y_train, y_train_proba))
        y_test_proba = qda.predict_proba(X_test)
        loss_test_qda.append(log_loss(y_test, y_test_proba))
        alpha_vals.append(a)
    except np.linalg.LinAlgError:
        continue

best_test_loss_idx = np.argmin(loss_test_qda)
print(f"Quadratic Discriminant Analysis:\n"
      f"    Train Loss: {loss_train_qda[best_test_loss_idx]:.4f} | "
      f"Test Loss: {loss_test_qda[best_test_loss_idx]:.4f} | "
      f"alpha: {alpha_vals[best_test_loss_idx]}")

plt.figure(figsize=(10, 6))
plt.plot(alpha_vals, loss_train_qda, c='r', ls='-', label="train")
plt.plot(alpha_vals, loss_test_qda, c='b', ls='--', label="test")
plt.xlabel("reg_param (alpha)")
plt.ylabel("loss")
plt.legend()
plt.tight_layout()
plt.show()


# naive bayes classifier
gnb = GaussianNB()
gnb.fit(X_train, y_train)
y_train_proba = gnb.predict_proba(X_train)
loss_train_gnb = log_loss(y_train, y_train_proba)
y_test_proba = gnb.predict_proba(X_test)
loss_test_gnb = log_loss(y_test, y_test_proba)

print(f"Naive Bayes Classifier:\n"
      f"    Train Loss: {loss_train_gnb:.4f} | "
      f"Test Loss: {loss_test_gnb:.4f}")


# logistic regression
lr = LogisticRegression(random_state=SEED)
lr.fit(X_train, y_train)

y_train_proba = lr.predict_proba(X_train)
loss_train_lr = log_loss(y_train, y_train_proba)
y_test_proba = lr.predict_proba(X_test)
loss_test_lr = log_loss(y_test, y_test_proba)

print(f"Logistic Regression:\n"
      f"    Train Loss: {loss_train_lr:.4f} | "
      f"Test Loss: {loss_test_lr:.4f}")


print("\nSUMMARY:")
print(f"KNN:   {loss_test_knn[best_test_loss_idx]:.4f}")
print(f"LDA:   {loss_test_lda:.4f}")
print(f"QDA:   {loss_test_qda[best_test_loss_idx]:.4f}")
print(f"NB:    {loss_test_gnb:.4f}")
print(f"LR:    {loss_test_lr:.4f}")