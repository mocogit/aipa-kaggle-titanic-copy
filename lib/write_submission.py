# -*- coding:utf8 -*-

import pandas as pd
import uuid

def _write(model, test, submission_df):
  predictions = model.predict(test)
  predictions = pd.DataFrame(predictions, columns=['Survived'])
  predictions = pd.concat((submission_df['PassengerId'], predictions), axis = 1)
  predictions.to_csv('submission_' + str(uuid.uuid1()) + '.csv', sep=",", index = False)
