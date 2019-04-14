# -*- coding:utf8 -*-

import pandas as pd
from sklearn.metrics import confusion_matrix


def create_correct(y, predict_y):
    return pd.DataFrame({'correct': y, 'predict': predict_y})


def matrix(y, predict_y):
    pre_core_df = create_correct(y, predict_y)
    print(confusion_matrix(pre_core_df['correct'], pre_core_df['predict']))


def cross(y, predict_y):
    # 正解した結果
    pre_core_df = create_correct(y, predict_y)
    # 正解だった
    _all_correct = pre_core_df[pre_core_df['correct'] == pre_core_df['predict']]['predict'].count()
     # 正解で生存していたを当てた
    _correct_survival = pre_core_df[(pre_core_df['correct'] == pre_core_df['predict']) & (1 == pre_core_df['predict'])]['predict'].count()
    # 正解で死亡を当てた
    _correct_dead = pre_core_df[(pre_core_df['correct'] == pre_core_df['predict']) & (0 == pre_core_df['predict'])]['predict'].count()
    # 生存と予測したけど、死亡してた
    _unexpected = pre_core_df[(pre_core_df['correct'] != pre_core_df['predict']) & (0 != pre_core_df['predict'])]['predict'].count()
    # 死亡と予測したけど、生存してた
    _missing = pre_core_df[(pre_core_df['correct'] != pre_core_df['predict']) & (1 != pre_core_df['predict'])]['predict'].count()

    print('正解: %d' % _all_correct)
    print('正解で生存を当てた: %d' % _correct_survival)
    print('正解で死亡を当てた: %d' % _correct_dead)
    print('予期しない: %d' % _unexpected)
    print('欠落した: %d'% _missing)
    print('間違い: %d' % (_unexpected + _missing))


# ランダムフォレスト用
# 効いている説明変数をpandasにして返す
def get_feature_importances(x, model):
    return pd.concat((pd.DataFrame(x.columns, columns = ['variable']), 
          pd.DataFrame(model.feature_importances_, columns = ['importance'])), 
          axis = 1).sort_values(by='importance', ascending = False)


def get_threat_score(x, y, predict_y):
    pre_core_df = create_correct(y, predict_y)
    correct = pre_core_df['correct']
    predict = pre_core_df['predict']

    tp = x.iloc[pre_core_df[(correct == predict) & (1 == predict)].index, ]
    tn = x.iloc[pre_core_df[(correct == predict) & (0 == predict)].index, ]
    fp = x.iloc[pre_core_df[(correct != predict) & (0 != predict)].index, ]
    fn = x.iloc[pre_core_df[(correct != predict) & (1 != predict)].index, ]

    print(tp['Survived'].count(), ':', tn['Survived'].count(), ':', fp['Survived'].count(), ':', fn['Survived'].count())

    return tp, tn, fp, fn
