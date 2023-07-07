import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.layers import Dense, Input # ANN
from tensorflow.keras.layers import LSTM, GRU, SimpleRNN # RNN
from tensorflow.keras.layers import Conv1D, MaxPooling1D, GlobalMaxPooling1D # CNN
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint

from sklearn.metrics import mean_absolute_percentage_error

np.random.seed(123)
tf.random.set_seed(123)

# !wget -nc https://lazyprogrammer.me/course_files/airline_passengers.csv

df = pd.read_csv('airline_passengers.csv', index_col='Month', parse_dates=True)

df['LogPassengers'] = np.log(df['Passengers'])

Ntest = 12
train = df.iloc[:-Ntest]
test = df.iloc[-Ntest:]

train_idx = df.index <= train.index[-1]
test_idx = df.index > train.index[-1]

df['DiffLogPassengers'] = df['LogPassengers'].diff()

# Make supervised dataset

series = df['DiffLogPassengers'].dropna().to_numpy()

T = 10
X = []
Y = []

for t in range(len(series) - T):
  x = series[t:t+T]
  X.append(x)
  y = series[t+T]
  Y.append(y)

# X = np.array(X).reshape(-1, T) # For ANN
X = np.array(X).reshape(-1, T, 1) # For CNN and RNN
Y = np.array(Y)
N = len(X)
print("X.shape", X.shape, "Y.shape", Y.shape)

Xtrain, Ytrain = X[:-Ntest], Y[:-Ntest]
Xtest, Ytest = X[-Ntest:], Y[-Ntest:]

# Basic ANN
i = Input(shape=(T,))
x = Dense(32, activation='relu')(i)

# RNN

i = Input(shape=(T, 1))
# can use SimpleRNN/GRU/LSTM
x = LSTM(32, return_sequences=True)(i) # default is tanh
x = LSTM(32)(x)
# when return_sequences=True, can use GlobalMaxPooling1D afterwards

# CNN (1D for time series, 2D for images)

i = Input(shape=(T, 1)) # single value time series
x = Conv1D(16, 3, activation='relu', padding='same')(i)
x = MaxPooling1D(2)(x)
x = Conv1D(32, 3, activation='relu', padding='same')(x)
x = GlobalMaxPooling1D()(x)

x = Dense(1)(x)
model = Model(i, x)

model.summary() # CNN and RNN (ANN?)

# change loss for classification and other tasks
model.compile(
    loss='mse',
    optimizer='adam',
    metrics='mae',
)

r = model.fit(
    Xtrain,
    Ytrain,
    epochs=100,
    validation_data=(Xtest, Ytest),
)

plt.plot(r.history['loss'], label='train loss')
plt.plot(r.history['val_loss'], label='test loss')
plt.legend()
plt.show()

train_idx[:T+1] = False # not predictable

Ptrain = model.predict(Xtrain).flatten()
Ptest = model.predict(Xtest).flatten()

df.loc[train_idx, 'Diff ANN Train Prediction'] = Ptrain
df.loc[test_idx, 'Diff ANN Test Prediction'] = Ptest

cols = ['DiffLogPassengers',
        'Diff Train Prediction',
        'Diff Test Prediction',]

df[cols].plot(figsize=(15, 5));

# Need to computer un-differenced predictions
df['ShiftLogPassengers'] = df['LogPassengers'].shift(1)
prev = df['ShiftLogPassengers']

# Last known train value
last_train = train.iloc[-1]['LogPassengers']

# 1-step forecast
df.loc[train_idx, '1step_train'] = prev[train_idx] + Ptrain
df.loc[test_idx, '1step_test'] = prev[test_idx] + Ptest

col2 = ['LogPassengers',
        '1step_train',
        '1step_test',]
df[col2].plot(figsize=(15, 5));

# multi-step forecast
multistep_predictions = []

# first test input
last_x = Xtest[0]

while len(multistep_predictions) < Ntest:
  # p = model.predict(last_x.reshape(1, -1))[0] # ANN
  p = model.predict(last_x.reshape(1, -1, 1))[0] # CNN

  # update the predictions list
  multistep_predictions.append(p)

  # make the new input
  last_x = np.roll(last_x, -1)
  last_x[-1] = p

df.loc[test_idx, 'multistep'] = last_train + np.cumsum(multistep_predictions)

col3 = ['LogPassengers',
        'multistep',
        '1step_test',]
df[col3].plot(figsize=(15, 5))

# make multi-output supervsied dataset
Tx = T
Ty = Ntest
X = []
Y = []

for t in range(len(series) - Tx - Ty + 1):
  x = series[t:t+Tx]
  X.append(x)
  y = series[t+Tx:t+Tx+Ty]
  Y.append(y)

# X = np.array(X).reshape(-1, Tx) # ANN
X = np.array(X).reshape(-1, Tx, 1) # CNN
Y = np.array(Y).reshape(-1, Ty)
N = len(X)
print("Y.shape", Y.shape, "X.shape", X.shape)

Xtrain_m, Ytrain_m = X[:-1], Y[:-1]
Xtest_m, Ytest_m = X[-1:], Y[-1:]

# Basic ANN
i = Input(shape=(Tx,))
x = Dense(32, activation='relu')(i)

# RNN
i = Input(shape=(Tx, 1))
x = LSTM(32, return_sequences=True)(i)
x = LSTM(32)(x)

# CNN
i = Input(shape=(Tx, 1))
x = Conv1D(16, 3, activation='relu', padding='same')(i)
x = MaxPooling1D(2)(x)
x = Conv1D(32, 3, activation='relu', padding='same')(i)
x = GlobalMaxPooling1D()(x)

x = Dense(Ty)(x)
model = Model(i, x)

model.summary()

model.compile(
    loss='mse',
    optimizer='adam',
    metrics='mae',
)

r = model.fit(
    Xtrain_m,
    Ytrain_m,
    epochs=100,
    validation_data=(Xtest_m, Ytest_m)
)

plt.plot(r.history['loss'], label='train loss')
plt.plot(r.history['val_loss'], label='test lsos')
plt.legend()
plt.show()

Ptrain = model.predict(Xtrain_m)
Ptest = model.predict(Xtest_m)
Ptrain.shape, Ptest.shape

Ptrain = Ptrain[:,0] # prediction for 1 stemp ahead (zeroth row)
Ptest = Ptest[0]

df.loc[test_idx, 'Diff Multi-Output Test Prediction'] = Ptest
col5 = ['DiffLogPassengers', 'Diff Multi-Output Test Prediction']
df[col5].plot(figsize=(15, 5));

df.loc[test_idx, 'multioutput'] = last_train + np.cumsum(Ptest)

col4 = ['LogPassengers', 'multistep', '1step_test', 'multioutput']
df[col4].plot(figsize=(15, 5));

# MAPE
test_log_pass = df.iloc[-Ntest:]['LogPassengers']
mape1 = mean_absolute_percentage_error(
    test_log_pass, df.loc[test_idx, 'multistep']
)
print("multi-step MAPE:", mape1)
mape2 = mean_absolute_percentage_error(
    test_log_pass, df.loc[test_idx, 'multioutput']
)
print("multi-output MAPE:", mape2)

