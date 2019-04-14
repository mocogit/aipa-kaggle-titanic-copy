# -*- coding:utf8 -*-

import pandas as pd


def get(path):
  train_df = pd.read_csv(path + '/train.csv')
  test_df = pd.read_csv(path + '/test.csv')
  submission_df = pd.read_csv(path + '/gender_submission.csv')
  return train_df, test_df, submission_df
