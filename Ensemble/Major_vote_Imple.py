from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from Iris_Dataset import X_train, y_train, X_test, y_test
from Majority_Vote_Classifier import MajorityVoteClassifier
import numpy as np

clf1 = LogisticRegression(penalty='l2',
                          C=0.001, random_state=0)
clf2 = DecisionTreeClassifier(max_depth=1,
                              criterion='entropy',
                              random_state=0)
clf3 = KNeighborsClassifier(n_neighbors=1,
                            p=2,
                            metric='minkowski')
pip1 = Pipeline([['sc', StandardScaler()], ['clf', clf1]])
pip3 = Pipeline([['sc', StandardScaler()], ['clf', clf3]])
clf_labels = ['Logistic Regression',
              'Decision Tree', 'KNN']
print('10-fold cross-validation:\n')
for clf, label in zip([pip1, clf2, pip3], clf_labels):
    scores = cross_val_score(estimator=clf,
                             X=X_train,
                             y=y_train, cv=10,
                             scoring='roc_auc')
    means = scores.mean()
    stds = scores.std()
    print("ROC AUC:%0.2f (+/- %0.2f) [%s]"
          % (means, stds, label))

mv_clf = MajorityVoteClassifier(classifiers=[pip1, clf2, pip3])
clf_labels += ['Majority Voting']
all_clf = [pip1, clf2, pip3, mv_clf]
print('Compare ensemble and individual\n')
for clf, label in zip(all_clf, clf_labels):
    scores = cross_val_score(estimator=clf,
                             X=X_train,
                             y=y_train, cv=10,
                             scoring='roc_auc')
    means = scores.mean()
    stds = scores.std()
    print("ROC AUC:%0.2f (+/- %0.2f) [%s]"
          % (means, stds, label))
