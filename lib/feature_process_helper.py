import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder


def names(train, test, submission_df):
    for i in [train, test]:
        i['Name_Len'] = i['Name'].apply(lambda x: len(x))
        i['Name_Title'] = i['Name'].apply(lambda x: x.split(',')[1]).apply(lambda x: x.split()[0])


    test = test.copy()
    test.insert(1, 'Survived', submission_df['Survived'])
    all = pd.concat([train, test], axis=0)
    all = all[train.columns]

    all['SameLastName'] = 0
    all.loc[all['Name'].apply(lambda x: x.split(',')[0]).duplicated() | all['Name'].apply(lambda x: x.split(',')[0]).duplicated(keep='last'), 'SameLastName'] = 1

    train = all.iloc[:len(train), :]
    test = all.iloc[len(train):, :]

    del train['Name']
    del test['Name']
    del test['Survived']

    return train, test


def age_null_flag(train, test):
    for i in [train, test]:
        i['Age_Null_Flag'] = i['Age'].apply(lambda x: 1 if pd.isnull(x) else 0)
    return train, test


def age_impute(train, test):
    # fori in [train, test]:
    #     i['Age_Null_Flag'] = i['Age'].apply(lambda x: 1 if pd.isnull(x) else 0)
    train['mean'] = train.groupby(['Name_Title', 'Pclass'])['Age'].transform('mean')
    train['Age'] = train['Age'].fillna(train['mean'])
    z = test.merge(train, on=['Name_Title', 'Pclass'], how='left').drop_duplicates(['PassengerId_x'])
    test['Age'] = np.where(test['Age'].isnull(), z['mean'], test['Age'])
    test['Age'] = test['Age'].fillna(test['Age'].mean())
    del train['mean']
    return train, test


def age_impute2(train, test):
    # for i in [train, test]:
    #     i['Age_Null_Flag'] = i['Age'].apply(lambda x: 1 if pd.isnull(x) else 0)
    train['mean'] = train.groupby(['Name_Title', 'Pclass'])['Age'].transform('mean')
    train['Age'] = train['Age'].fillna(train['mean'])
    z = test.merge(train, on=['Name_Title', 'Pclass'], how='left').drop_duplicates(['PassengerId_x'])
    test['Age'] = np.where(test['Age'].isnull(), z['mean'], test['Age'])
    test['Age'] = test['Age'].fillna(test['Age'].mean())
    del train['mean']
    return train, test


# Pclassと敬称毎にグループ化した中央値をとってみる
def age_impute3(train, test):
    for i in [train, test]:
        not_null_age = i.loc[i['Age'].isnull() == False, :]
        median = not_null_age.groupby(['Pclass', 'Name_Title'])[['Age']].median().reset_index()

        def grouping_pclass_name_title_median(x):
                _pclass = median[median['Pclass'] == x['Pclass']]
                _is_name_title = _pclass[_pclass['Name_Title'] == x['Name_Title']]
                if _is_name_title.empty == False:
                    return int(_is_name_title['Age'])
                return 0

        i.loc[i['Age'].isnull() == True, 'Age'] = i.loc[i['Age'].isnull() == True, ['Pclass', 'Name_Title']].apply(grouping_pclass_name_title_median, axis=1)
    return train, test


def cabin_null_flag(train, test):
    for i in [train, test]:
        i['Cabin_Null_Flag'] = np.where(i['Cabin'].isnull(), 1, 0)
    return train, test


def cabin_count(train, test):
    for i in [train, test]:
        i['CabinCount'] = pd.Series()
        i.loc[i['Cabin'].isnull() == False, 'CabinCount'] = i[i['Cabin'].isnull() == False]['Cabin'].apply(lambda x: len(x.split(' ')))
        i.loc[i['Cabin'].isnull() == True, 'CabinCount'] = 0
        i.loc[:, 'CabinCount'] = i['CabinCount'].astype(int)
    return train, test


def ticket_dummies(train, test):
    for i in [train, test]:
        i['Ticket_Prefix'] = pd.Series()
        i['Ticket_Prefix'] = i[i['Ticket'].str.match('[A-z]')]['Ticket'].apply(lambda x: x.split(' ')[0])
        i['Ticket_Prefix'] = i['Ticket_Prefix'].fillna('Nothing')
    return train, test


def same_ticket_grouping(train, test, submission_df):
    test = test.copy()
    test.insert(1, 'Survived', submission_df['Survived'])
    all = pd.concat([train, test], axis=0)
    all = all[train.columns]
    duplicate_sex_count = all[all['Ticket'].duplicated() | all['Ticket'].duplicated(keep='last')] \
        .groupby(['Ticket', 'Sex'])[['PassengerId']] \
        .count()
    duplicate_sex_count_reset_index = duplicate_sex_count.reset_index()
    duplicate_ticket_sex = duplicate_sex_count_reset_index.pivot_table(index='Ticket', columns='Sex',
                                                                       values='PassengerId', aggfunc='count', fill_value=0).reset_index()
    male_only = duplicate_ticket_sex[(duplicate_ticket_sex['female'] == 0) & (duplicate_ticket_sex['male'] == 1)]
    female_only = duplicate_ticket_sex[(duplicate_ticket_sex['female'] == 1) & (duplicate_ticket_sex['male'] == 0)]
    male_female = duplicate_ticket_sex[(duplicate_ticket_sex['female'] == 1) & (duplicate_ticket_sex['male'] == 1)]

    all['SameTicket'] = pd.Series()
    for k, v in {'Only_Male': male_only, 'Only_Female': female_only, 'Male_Female': male_female}.items():
        all.loc[train['Ticket'].isin(v['Ticket'].tolist()), 'SameTicket'] = k

    all.loc[:, 'SameTicket'] = all['SameTicket'].fillna('Not_Same')
    train = all.iloc[:len(train), :]
    test = all.iloc[len(train):, :]
    del test['Survived']
    return train, test


def age_class(train, test):
    for i in [train, test]:
        i['AgeClass'] = (i['Age'] // 10 * 10).astype(int)
    return train, test


def cabin(train, test):
    for i in [train, test]:
        i['Cabin_Letter'] = i['Cabin'].apply(lambda x: str(x)[0])
        del i['Cabin']
    return train, test


def embarked_impute(train, test):
    for i in [train, test]:
        i['Embarked'] = i['Embarked'].fillna('S')
    return train, test


# ticketの重複を数える関数を作成
def apply_ticket_count(train, test):
    for i in [train, test]:
        _duplicated = i[i['Ticket'].duplicated() | i['Ticket'].duplicated(keep='last')].groupby(['Ticket'], as_index=False)[['PassengerId']].count()
        def replace_ticket_duplicated_count(_series):
            _is = _duplicated[_series == _duplicated['Ticket']]['PassengerId']
            if _is.empty == False:
                return int(_is)
            return 0
        i['TicketCount'] = i['Ticket'].apply(replace_ticket_duplicated_count)
    return train, test


def dummies(train, test, columns = ['Pclass', 'Sex', 'Embarked', 'Cabin_Letter', 'Name_Title']):
    for column in columns:
        train[column] = train[column].apply(lambda x: str(x))
        test[column] = test[column].apply(lambda x: str(x))
        good_cols = [column+'_'+i for i in train[column].unique() if i in test[column].unique()]
        train = pd.concat((train, pd.get_dummies(train[column], prefix = column)[good_cols]), axis = 1)
        test = pd.concat((test, pd.get_dummies(test[column], prefix = column)[good_cols]), axis = 1)
        del train[column]
        del test[column]
    return train, test

def drop(train, test, bye = ['Ticket', 'SibSp', 'Parch']):
    for i in [train, test]:
        for z in bye:
            del i[z]
    return train, test


def family_size_int(train, test):
  for i in [train, test]:
    i['FamilySize'] = i.Parch + i.SibSp + 1
  return train, test


def add_last_name(train, test):
  for i in [train, test]:
    i['LastName'] = i['Name'].str.split(',').str[0]
  return train, test


def fam_size(train, test):
    for i in [train, test]:
        i['Fam_Size'] = np.where((i['SibSp']+i['Parch']) == 0 , 'Solo',
                           np.where((i['SibSp']+i['Parch']) <= 3,'Nuclear', 'Big'))
        # del i['SibSp']
        # del i['Parch']
    return train, test


# def fare_length(train, test):
#     for i in [train, test]:
#         i['Fare_Length'] = i['Fare'].apply(lambda x: len(x))
#     return train, test


def lda(X_train, X_test, y_train, cols=['Age', 'Fare']):
    sc = StandardScaler()
    X_train_std = sc.fit_transform(X_train[cols])
    X_test_std = sc.transform(X_test[cols])
    lda = LDA(n_components=None)
    X_train_lda = lda.fit_transform(X_train_std, y_train.values.ravel())
    X_test_lda = lda.transform(X_test_std)
    X_train = pd.concat((X_train, pd.DataFrame(X_train_lda)), axis=1)
    X_test = pd.concat((X_test, pd.DataFrame(X_test_lda)), axis=1)
    for i in cols:
        del X_train[i]
        del X_test[i]
    return X_train, X_test

def titles_grouped(train, test):
    for i in [train, test]:
        i['Name_Title'] = np.where((i['Name_Title']).isin(['Mr.', 'Mrs.', 'Miss.', 'Master.', 'Dr.']),
                                   i['Name_Title'], 'other')
    return train, test


def ticket_length(train, test):
    for i in [train, test]:
        i['Ticket_Len'] = i['Ticket'].apply(lambda x: len(x))
    return train, test


def ticket_grouped(train, test):
    for i in [train, test]:
        i['Ticket_Lett'] = i['Ticket'].apply(lambda x: str(x)[0])
        i['Ticket_Lett'] = i['Ticket_Lett'].apply(lambda x: str(x))
        i['Ticket_Lett'] = np.where((i['Ticket_Lett']).isin(['1', '2', '3', 'S', 'P', 'C', 'A']), i['Ticket_Lett'],
                                   np.where((i['Ticket_Lett']).isin(['W', '4', '7', '6', 'L', '5', '8']),
                                            'Low_ticket', 'Other_ticket'))
        i['Ticket_Len'] = i['Ticket'].apply(lambda x: len(x))
        del i['Ticket']
    return train, test


def ticket_grouped2(train, test):
    for i in [train, test]:
        i['Ticket_Lett'] = i['Ticket'].apply(lambda x: str(x)[0])
        i['Ticket_Lett'] = i['Ticket_Lett'].apply(lambda x: str(x))
        i['Ticket_Lett'] = np.where((i['Ticket_Lett']).isin(['1', '2', '3', 'S', 'P', 'C', 'A']), i['Ticket_Lett'],
                                   np.where((i['Ticket_Lett']).isin(['W', '4', '7', '6', 'L', '5', '8']),
                                            'Low_ticket', 'Other_ticket'))
        i['Ticket_Len'] = i['Ticket'].apply(lambda x: len(x))
        del i['Ticket']
    return train, test


def cabin_num(train, test):
    for i in [train, test]:
        i['Cabin_num1'] = i['Cabin'].apply(lambda x: str(x).split(' ')[-1][1:])
        i['Cabin_num1'].replace('an', np.NaN, inplace = True)
        i['Cabin_num1'] = i['Cabin_num1'].apply(lambda x: int(x) if not pd.isnull(x) and x is not '' else np.NaN)
    train['Cabin_num'], bins = pd.qcut(train['Cabin_num1'],3, retbins=True)
    test['Cabin_num'] = pd.cut(test['Cabin_num1'], bins=bins, include_lowest=True)
        #i['Cabin_num'] = i['Cabin_num'].isnull().apply(lambda x: float(x))
    train = pd.concat((train, pd.get_dummies(train['Cabin_num'], prefix = 'Cabin_num')), axis = 1)
    test = pd.concat((test, pd.get_dummies(test['Cabin_num'], prefix = 'Cabin_num')), axis = 1)
    del train['Cabin_num']
    del test['Cabin_num']
    del train['Cabin_num1']
    del test['Cabin_num1']
    return train, test


def cabin_fillna(train, test, submission_df):
    test_copy = test.copy()
    test_copy.insert(1, 'Survived', submission_df['Survived'])
    all = pd.concat([train, test_copy], axis=0)
    all = all[train.columns]
    all.loc[all['Cabin'].isnull() == False, 'CabinPrefix'] = all.loc[all['Cabin'].isnull() == False, 'Cabin'].apply(lambda x: x[0])
    all.loc[all['Fare'].isnull() == True, 'Fare'] = all['Fare'].median()

    x = all.loc[all['Cabin'].isnull() == False, ['Pclass', 'Fare']]
    all['Name_Length'] = all['Name'].apply(lambda x: len(x))
    x['Name_Length'] = all.loc[all['Cabin'].isnull() == False, 'Name_Length']
    _y = all.loc[all['Cabin'].isnull() == False, 'CabinPrefix']

    label = LabelEncoder()
    label.fit(_y)
    y = label.transform(_y)

    search_parameter = {
      'criterion': 'entropy',
      'min_samples_leaf': 1,
      'min_samples_split': 2,
      'n_estimators': 100
    }
    rf = RandomForestClassifier(max_features='auto',
                                    oob_score=True,
                                    random_state=1,
                                    n_jobs=-1)
    rf.set_params(**search_parameter)
    rf.fit(x, y)

    all['CabinPrefix'] = label.inverse_transform(rf.predict(all[['Pclass', 'Fare', 'Name_Length']]))
    del all['Name_Length']
    
    train = all.iloc[:len(train), :]
    test = all.iloc[len(train):, ]
    del test['Survived']

    return train, test
