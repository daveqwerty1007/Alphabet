import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, confusion_matrix
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

path = '/Users/zhaojianbo/Desktop/Alphabet/Week1/creditcard.csv'
data = pd.read_csv(path)


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

## model setup
model = Sequential()
model.add(Dense(32, input_dim=train_X.shape[1], activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(train_X, train_y, epochs=10, batch_size=32, validation_split=0.1)

# Evaluate the model on the test set
test_loss, test_accuracy = model.evaluate(test_X, test_y)
print("Neural Network Test Accuracy:", test_accuracy)

# Make predictions on the test set
nn_predictions = (model.predict(test_X) > 0.5).astype("int32")

# Evaluate the model
nn_accuracy = accuracy_score(test_y, nn_predictions)
print("Neural Network Accuracy:", nn_accuracy)
