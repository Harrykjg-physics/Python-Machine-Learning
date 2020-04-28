from Major_vote_Imple import mv_clf
from sklearn.model_selection import GridSearchCV
from Iris_Dataset import X_train, y_train

# print(mv_clf.get_params())
params = {'decisiontreeclassifier__max_depth': [1, 2],
          'pipeline-1__clf__C': [0.001, 0.1, 100]}
grid = GridSearchCV(estimator=mv_clf, param_grid=params,
                    cv=10, scoring='roc_auc')
grid.fit(X_train, y_train)

# for params, mean_score, scores in grid.grid_scores_:
#     print("%0.3f +/- %0.2f %r"
#           % (mean_score, scores.std()/2, params))

print(grid.best_params_)
print(grid.best_score_)
