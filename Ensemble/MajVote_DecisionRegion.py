from sklearn.preprocessing import StandardScaler
from Iris_Dataset import X_train, y_train
from itertools import product
import numpy as np
import matplotlib.pyplot as plt
from Major_vote_Imple import all_clf, clf_labels

sc = StandardScaler()
X_train_std = sc.fit_transform(X_train)
xmin = X_train_std[:, 0].min() - 1
xmax = X_train_std[:, 0].max() + 1
ymin = X_train_std[:, 1].min() - 1
ymax = X_train_std[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(xmin, xmax, 0.1),
                     np.arange(ymin, ymax, 0.1))
f, axarr = plt.subplots(nrows=2, ncols=2,
                        sharex='col',
                        sharey='row',
                        figsize=(7, 5))
for idx, clf, tt in zip(product([0, 1], [0, 1]),
                        all_clf, clf_labels):
    clf.fit(X_train_std, y_train)
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    axarr[idx[0], idx[1]].contourf(xx, yy, Z, alpha=0.3)
    axarr[idx[0], idx[1]].scatter(X_train_std[y_train == 0, 0]
                                  , X_train_std[y_train == 0, 1],
                                  c='blue', marker='^', s=50)
    axarr[idx[0], idx[1]].scatter(X_train_std[y_train == 1, 0]
                                  , X_train_std[y_train == 1, 1],
                                  c='red', marker='o', s=50)
    axarr[idx[0], idx[1]].set_title(tt)

plt.text(-1.5, -2.5, s='Sepal Width[standardized]',
         ha='center', va='center_baseline', fontsize=12)
plt.text(-4.5, 0.5, s='Petal Length[standardized]',
         ha='left', va='center', fontsize=12, rotation=90)
plt.show()




