import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense, Dropout

df = pd.read_csv('"C:\\Users\\ad\Downloads\\archive (1)\\Housing.csv"')


X = df[['area', 'bedrooms', 'bathrooms', 'mainroad', 'guestroom']]  
y = df['price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=45, random_state=42)


scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = Sequential()
model.add(Dense(32, activation='relu', input_dim=X_train.shape[1]))
model.add(Dropout(0.2))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1, activation='linear'))

model.compile(loss='mean_squared_error', optimizer='adam')
 
model.fit(X_train, y_train, epochs=100, batch_size=32)
loss = model.evaluate(X_test, y_test)
print("Sai số trên tập test: %.2f" % loss)
y_pred = model.predict(X_test)

import matplotlib.pyplot as plt
plt.scatter(y_test, y_pred)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=4)
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.show()




