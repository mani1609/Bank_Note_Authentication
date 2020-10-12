import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
from sklearn import metrics
%matplotlib inline
import warnings
from sklearn.exceptions import DataConversionWarning
warnings.filterwarnings(action='ignore')
data = pd.read_csv('bank_note_data.csv',
    names = ['variance','skewness','curtosis','entropy','class'])

data.head(3)
data.describe()
data.shape
data.isna().any()
data.dtypes
data['class'].unique()
sns.countplot(x='class', data= data)
sns.violinplot( y=data['curtosis'])
sns.violinplot( y=data['entropy'])
sns.violinplot( y=data['variance'])
sns.violinplot( y=data['skewness'])
p1=sns.kdeplot(data['curtosis'], shade=True, color="r")
p1=sns.kdeplot(data['variance'], shade=True, color="b")
sns.jointplot(x=data['curtosis'], y=data['entropy'], kind='hex', linewidth = 2)
sns.jointplot(x=data['skewness'], y=data['variance'], kind='hex', color = 'skyblue', linewidth = 2)
sns.jointplot(x=data['curtosis'], y=data['variance'], kind='hex', linewidth = 2)
X = data[['variance', 'skewness' ,'curtosis', 'entropy']]
y = data[['class']]

from sklearn.model_selection import train_test_split # Support Vector Machine
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

from sklearn.svm import SVC
SVC()
svc = SVC()

# Creating a dictionary of parameters

parameters = {
    'kernel' : ['linear','rbf'],
    'C' : [0.1,1,5, 7, 10,15]
}
gridCV = GridSearchCV(svc, parameters, cv=5)

# Retraining the model with optimum features

gridFit =gridCV.fit(X_train,y_train.values.ravel())
gridFit
gridFit.best_estimator_

def printBestParameters(inputFit):
    print ('Best Parameters: {} \n'.format(inputFit.best_params_))
printBestParameters(gridFit)

 # Predecting new note

newNote = np.array([4,7,-2.797,-0.55])
newNote = newNote.reshape(1,-1)
gridFit.predict(newNote)