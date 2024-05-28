import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, confusion_matrix


data = pd.read_csv('/Users/zhaojianbo/Desktop/creditcard.csv')


##print(data.head())

X = data.drop('Class', axis=1)
y = data['Class']

train_X = X[:-100]
train_y = y[:-100]
test_X = X[-100:]
test_y = y[-100:]

model = LinearRegression().fit(train_X, train_y)

##print("Intercept:", model.intercept_)
##print("Coefficients:", model.coef_)

# Make predictions on the test set
predictions = model.predict(test_X)

predictions_binary = np.round(predictions)

accuracy = accuracy_score(test_y, predictions_binary)
print("Accuracy:", accuracy)

conf_matrix = confusion_matrix(test_y, predictions_binary)
print("Confusion Matrix:")
print(conf_matrix)


## 感知机
from sklearn.linear_model import Perceptron

ppn = Perceptron(tol=1e-3, random_state=0)

ppn.fit(X, y)

##print("权重:", ppn.coef_)
##print("偏置:", ppn.intercept_)

predictions = ppn.predict(X)
print("预测:", predictions)
print("accuracy:", accuracy_score(y, ppn.predict(X)))

## Nerual