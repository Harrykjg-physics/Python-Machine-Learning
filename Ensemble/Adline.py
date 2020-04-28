import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from Decision_Region import plot_decision_regions


class AdlineGD:

    def __init__(self, eta=0.01, n_iter=50):
        self.eta = eta
        self.n_iter = n_iter

    def fit(self, X, y):
        self.w_ = np.zeros(1 + X.shape[1])
        self.cost_ = []
        for _ in range(self.n_iter):
            output = self.net_input(X)
            errors = y - output
            self.w_[1:] += self.eta * X.T.dot(errors)
            self.w_[0] = self.eta * errors.sum()
            cost = (errors**2).sum() / 2.0
            self.cost_.append(cost)
        return self

    def net_input(self, X):
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def activation(self, X):
        return self.net_input(X)

    def predict(self, X):
        return np.where(self.activation(X) >= 0.0, 1, -1)


df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-'
                 'databases/iris/iris.data', header=None)

y = df.iloc[0:100, 4].values
y = np.where(y == 'Iris-setosa', -1, 1)
X = df.iloc[0:100, [0, 2]].values

fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(8, 4))
ada1 = AdlineGD(eta=0.01, n_iter=10).fit(X, y)
ax[0].plot(range(1, len(ada1.cost_) + 1), np.log10(ada1.cost_),
           marker='o')
ax[0].set_xlabel('Epoches')
ax[0].set_ylabel('log(SSE)')
ax[0].set_title('Adline - learning rate 0.01')
ada2 = AdlineGD(eta=0.0001, n_iter=10).fit(X, y)
ax[1].plot(range(1, len(ada2.cost_) + 1), np.log10(ada2.cost_),
           marker='x')
ax[0].set_xlabel('Epoches')
ax[0].set_ylabel('log(SSE)')
ax[0].set_title('Adline - learning rate 0.0001')
plt.show()

X_std = np.copy(X)
X_std[:, 0] = (X[:, 0] - X[:, 0].mean())/X[:, 0].std()
X_std[:, 1] = (X[:, 1] - X[:, 1].mean())/X[:, 1].std()

ada = AdlineGD(eta=0.01, n_iter=15)
ada.fit(X_std, y)

plot_decision_regions(X_std, y, classifier=ada)
plt.title('Adline-Gradient Descent')
plt.xlabel('sepal length(standardized)')
plt.ylabel('pedal length(standardized)')
plt.legend(loc='upper left')
plt.show()
plt.plot(range(1, len(ada.cost_) + 1), ada.cost_, marker='o')
plt.xlabel('Epoches')
plt.ylabel('SSE')
plt.show()
