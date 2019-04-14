# -*- coding:utf8 -*-
from sklearn.ensemble import RandomForestClassifier
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import cross_val_score
import numpy as np


def search(train, test):
  rf = RandomForestClassifier(max_features='auto',
                                  oob_score=True,
                                  random_state=1,
                                  n_jobs=-1)

  param_grid = { "criterion"   : ["gini", "entropy"],
               "min_samples_leaf" : [1,5,10],
               "min_samples_split" : [2, 4, 10, 12, 16],
               "n_estimators": [50, 100, 400, 700, 1000]}

  gs = GridSearchCV(estimator=rf,
                    param_grid=param_grid,
                    scoring='accuracy',
                    cv=3,
                    n_jobs=-1)
  return gs.fit(train, test)


def learn(train, test, parameter):
  rf = RandomForestClassifier()
  rf.set_params(**parameter)
  return rf.fit(train, test)


def validation(train, test, model):
  cv_scores = cross_val_score(model, train, test, cv=10, n_jobs=-1)
  print('CV accuracy: %.3f +/- %.3f' % (np.mean(cv_scores), np.std(cv_scores)))
