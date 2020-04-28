from sklearn import datasets
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score
from Decision_Region import plot_decision_regions
import matplotlib.pyplot as plt


iris = datasets.load_iris()
X = iris.data[:, [2, 3]]
y = iris.target

X_train, X_test, y_train, y_test = \
    train_test_split(X, y, test_size=0.3, random_state=0)
print(len(X_train))
print(len(X_test))

sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

ppn = Perceptron(eta0=0.1, n_iter_no_change=40, random_state=0)
ppn.fit(X_train_std, y_train)
y_pre = ppn.predict(X_test_std)
print('Misclassified samples:%d' % (y_test != y_pre).sum())
print('Accuracy: %.2f' % accuracy_score(y_test, y_pre))

# vstack 表示垂直方向堆积数列
# hstack 表示水平方向堆积数列
# 也就是将交叉验证的结果恢复回来

X_combined_std = np.vstack((X_train_std, X_test_std))
y_combined = np.hstack((y_train, y_test))
plot_decision_regions(X=X_combined_std, y=y_combined, classifier=ppn,
                      test_idx=range(105, 150))
plt.xlabel('petal length [standardized]')
plt.ylabel('petal width [standardized]')
plt.legend(loc='upper left')
plt.show()
