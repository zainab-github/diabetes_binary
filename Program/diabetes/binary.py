import matplotlib.pyplot as plt
import sklearn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.neural_network import MLPClassifier
from sklearn import preprocessing

train_data = pd.read_csv("diabetes_risk_prediction_dataset.csv")

print(train_data.head())

pre_process = preprocessing.LabelEncoder()

train_data['class'] = pre_process.fit_transform(train_data['class'])

train_data['Age'] = pre_process.fit_transform(train_data['Age'])
train_data['Gender'] = pre_process.fit_transform(train_data['Gender'])
train_data['Polyuria'] = pre_process.fit_transform(train_data['Polyuria'])
train_data['Polydipsia'] = pre_process.fit_transform(train_data['Polydipsia'])
train_data['sudden weight loss'] = pre_process.fit_transform(train_data['sudden weight loss'])
train_data['weakness'] = pre_process.fit_transform(train_data['weakness'])
train_data['Polyphagia'] = pre_process.fit_transform(train_data['Polyphagia'])
train_data['Genital thrush'] = pre_process.fit_transform(train_data['Genital thrush'])
train_data['visual blurring'] = pre_process.fit_transform(train_data['visual blurring'])
train_data['Itching'] = pre_process.fit_transform(train_data['Itching'])
train_data['Irritability'] = pre_process.fit_transform(train_data['Irritability'])
train_data['delayed healing'] = pre_process.fit_transform(train_data['delayed healing'])
train_data['partial paresis'] = pre_process.fit_transform(train_data['partial paresis'])
train_data['muscle stiffness'] = pre_process.fit_transform(train_data['muscle stiffness'])
train_data['Alopecia'] = pre_process.fit_transform(train_data['Alopecia'])
train_data['Obesity'] = pre_process.fit_transform(train_data['Obesity'])

x_train = train_data[['Age', 'Gender', 'Polyuria', 'Polydipsia', 'sudden weight loss', 'weakness', 'Polyphagia',
                      'Genital thrush', 'visual blurring', 'Itching', 'Irritability', 'delayed healing',
                      'partial paresis', 'muscle stiffness', 'Alopecia', 'Obesity']]

y_train = train_data['class']

x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.3, random_state=0)

classifier = MLPClassifier(hidden_layer_sizes=(6, 5), random_state = 0, verbose=False, learning_rate_init=0.01)

classifier.fit(x_train, y_train)

y_pred = classifier.predict(x_test)

print('\n Accuracy: {}%'.format(round(metrics.accuracy_score(y_test, y_pred), 2)*100))
print('\n Precision: {}%'.format(round(metrics.precision_score(y_test, y_pred,), 2)*100))
print('\n Recall: {}%'.format(round(metrics.recall_score(y_test, y_pred), 2)*100))