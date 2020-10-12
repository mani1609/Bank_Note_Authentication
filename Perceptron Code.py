import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
import tensorflow.contrib.learn as learn
from sklearn.metrics import classification_report,confusion_matrix

data = pd.read_csv('bank_note_data.csv')

scaler = StandardScaler()
scaler.fit(data.drop('Class',axis=1))
scaled_feat = scaler.transform(data.drop('Class',axis=1))

# Convert the scaled features to a new dataframe.
scaled_data = pd.DataFrame(data=scaled_feat,columns=['Image.Var', 'Image.Skew', 'Image.Curt', 'Entropy'])

# Train Test Split
X = scaled_data
y = data['Class']

X = X.as_matrix()
y = y.as_matrix()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

# Contrib.learn

feature_columns = tf.contrib.learn.infer_real_valued_columns_from_input(X_train)
classifier = learn.DNNClassifier(feature_columns=feature_columns,hidden_units=[10,20,10],n_classes=3)

classifier.fit(X_train,y_train,steps=200,batch_size=20)

# Model Evaluation
predictions = list(classifier.predict(X_test))

print(confusion_matrix(y_test,predictions))
print('\n')
print(classification_report(y_test,predictions))