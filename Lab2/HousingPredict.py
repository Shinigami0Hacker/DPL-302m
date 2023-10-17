import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

df = pd.read_csv("./Housing.csv")

train_label = ['area', 'bedrooms', 'bathrooms', 'stories', 'parking']
X = df.loc[:, train_label][:500].to_numpy()
Y = df.loc[:, "price"][:500].to_numpy()

x_train, x_val, y_train, y_val = train_test_split(X, Y, test_size=0.1, shuffle=True)

test = df.tail(45).to_numpy()

model = tf.keras.Sequential([
    tf.keras.layers.Dense(5, activation='relu'),
    tf.keras.layers.Dense(3, activation='relu'),
    tf.keras.layers.Dense(1)
])


model.compile(optimizer='adam', loss='mse')

history = model.fit(x=x_train, y=y_train, validation_data=(x_val, y_val), batch_size=12, epochs=64)

prediction = model.predict(x_val)
print(prediction)
