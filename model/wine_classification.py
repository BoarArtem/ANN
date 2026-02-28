import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

data = pd.read_csv("../data/WineQT.csv")
data = data.drop(['Id'], axis=1)

X = data.drop(['quality'], axis=1)
y = data['quality']
y = y - 3

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1], )), # Hidden layer 1
    Dense(64, activation='relu'), # Hidden layer 2

    Dense(6, activation='softmax') # Output layer
])

opt=tf.keras.optimizers.Adam(learning_rate=0.001)
loss=tf.keras.losses.SparseCategoricalCrossentropy()

model.compile(optimizer=opt, loss=loss, metrics=["accuracy"])
model.fit(X_train, y_train, epochs=100, validation_data=(X_test, y_test))

print(model.summary())
model.save('wine.h5')
