from sklearn.model_selection import GridSearchCV
from pipeline import pipe_svc
from sklearn.tree import DecisionTreeClassifier
from Breast_Cancer_Dataset import X_train, y_train, X_test, y_test

# SVM

param_range = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
param_grid = [{'clf__C': param_range, 'clf__kernel': ['linear']},
              {'clf__C': param_range, 'clf__gamma': param_range,
               'clf__kernel': ['rbf']}]
gs_svm = GridSearchCV(estimator=pipe_svc, param_grid=param_grid,
                      scoring='accuracy', cv=10, n_jobs=-1)
# gs_svm = gs_svm.fit(X_train, y_train)
# print(gs_svm.best_score_)
# print(gs_svm.best_params_)

# clf = gs_svm.best_estimator_
# clf.fit(X_train, y_train)
# print('Test Accuracy: %.3f' % clf.score(X_test, y_test))

# Decision Tree

gs_dt = \
    GridSearchCV(estimator=DecisionTreeClassifier(random_state=0),
                 param_grid=[{'max_depth':
                                  [1, 2, 3, 4, 5, 6, 7, None]}],
                 scoring='accuracy',
                 cv=5)



