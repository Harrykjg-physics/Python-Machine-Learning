from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from Major_vote_Imple import all_clf, clf_labels
from Iris_Dataset import X_train, y_train, X_test, y_test
import matplotlib.pyplot as plt

colors = ['black', 'orange', 'blue', 'green']
linestyle = [':', '--', '-.', '-']
for clf, label, clr, ls in zip(all_clf,
                               clf_labels,
                               colors, linestyle):
    y_pred = clf.fit(X_train, y_train).predict_proba(X_test)[:, 1]
    fpr, tpr, thresholds = \
        roc_curve(y_true=y_test, y_score=y_pred)
    roc_auc = auc(x=fpr, y=tpr)
    plt.plot(fpr, tpr, color=clr, linestyle=ls,
             label='%s (auc = %0.2f)' % (label, roc_auc))
plt.legend(loc='lower right')
plt.plot([0, 1], [0, 1], linestyle='--', color='grey',
         linewidth=2)
plt.xlim([-0.1, 1.1])
plt.ylim([-0.1, 1.1])
plt.grid()
plt.xlabel('fpr')
plt.ylabel('tpr')
plt.show()



