
# coding: utf-8

# In[1]:

import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score
import csv as csv


# In[2]:

train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')


# In[3]:

sex_encoder = preprocessing.LabelEncoder()
sex_encoder.fit(train['Sex'])

train['male'] = sex_encoder.transform(train['Sex'])
test['male'] = sex_encoder.transform(test['Sex'])


# In[4]:

train_embarked_dummied = pd.get_dummies(train["Embarked"], prefix='embarked', drop_first=True)
test_embarked_dummied = pd.get_dummies(test["Embarked"], prefix='embarked', drop_first=True)

train = pd.concat([train, train_embarked_dummied], axis=1)
test = pd.concat([test, test_embarked_dummied], axis=1)


# In[5]:

features = ['Age','SibSp','Parch','Fare','male','Pclass','embarked_Q','embarked_S']


# In[6]:

X_train = train[features]
X_test = test[features]

y_train = train['Survived']


# In[7]:

imputer = preprocessing.Imputer(missing_values='NaN', strategy='mean', axis=0)


# In[8]:

imputer.fit(X_train)

X_train = imputer.transform(X_train)
X_test = imputer.transform(X_test)


# In[9]:

parameter_grid = dict(n_estimators=list(range(1, 5001, 1000)),
                      criterion=['gini','entropy'],
                      max_features=list(range(1, len(features), 2)),
                      max_depth= [None] + list(range(5, 25, 1)))


# In[10]:

random_forest = RandomForestClassifier(random_state=0, n_jobs=-1)


clf = GridSearchCV(estimator=random_forest, param_grid=parameter_grid, cv=5, verbose=2, n_jobs=-1)


cv_scores = cross_val_score(clf, X_train, y_train)
print(cv_scores)
print(np.mean(cv_scores))

clf.fit(X_train, y_train)
predictions = clf.predict(X_test).astype(int)

ids = test['PassengerId'].values

predictions_file = open("predictions.csv", "w")
open_file_object = csv.writer(predictions_file)
open_file_object.writerow(["PassengerId","Survived"])
open_file_object.writerows(zip(ids, predictions))
predictions_file.close()
print('Done!')
