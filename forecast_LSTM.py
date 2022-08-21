# -*- coding: utf-8 -*-
"""
Created on Mon Mar 28 16:05:03 2022

@author: Stas
"""
import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
import pandas as pd
import seaborn as sns
from pylab import rcParams
import matplotlib.pyplot as plt
from matplotlib import rc
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import Dropout, Activation, Dense, LSTM, Input
from tensorflow.keras import optimizers
from tensorflow.python.keras.layers import CuDNNLSTM
from tensorflow.keras.models import Model

#начальные параметры
sns.set(style='whitegrid', palette='muted', font_scale=1.5)
rcParams['figure.figsize'] = 14, 8
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

#Data preparation
csv_path = "data/Binance_BTCUSDT_d.csv"

df = pd.read_csv(csv_path, parse_dates=['date'])
df = df.sort_values('date')
df.reset_index(drop=True,inplace=True)
#df.shape

#Plotting initial data
# ax = df.plot(x='date', y='close');
# ax.set_xlabel("Date")
# ax.set_ylabel("Close Price (USD)")

#data normalization
scaler = MinMaxScaler()
close_price = df.close.values.reshape(-1, 1)
scaled_close = scaler.fit_transform(close_price)

#Check if there any nan values
#np.isnan(scaled_close).any()

scaled_close = scaled_close[~np.isnan(scaled_close)]
scaled_close = scaled_close.reshape(-1, 1)
np.isnan(scaled_close).any()

#Preprocessing
SEQ_LEN = 50

def to_sequences(data, seq_len):
    d = []

    for index in range(len(data) - seq_len):
        d.append(data[index: index + seq_len])

    return np.array(d)

def preprocess(data_raw, seq_len, train_split):

    data = to_sequences(data_raw, seq_len)

    num_train = int(train_split * data.shape[0])

    X_train = data[:num_train, :-1, :]
    y_train = data[:num_train, -1, :]

    X_test = data[num_train:, :-1, :]
    y_test = data[num_train:, -1, :]

    return X_train, y_train, X_test, y_test


X_train, y_train, X_test, y_test = preprocess(scaled_close, SEQ_LEN, train_split = 0.95)

#X_train.shape
#X_test.shape 

#Model
WINDOW_SIZE = SEQ_LEN - 1

lstm_input = Input(shape=(WINDOW_SIZE, X_train.shape[-1]), name='lstm_input')
x = LSTM(50, return_sequences=True, name='lstm_0')(lstm_input)
x = LSTM(50, return_sequences=True, name='lstm_1')(x)
x = LSTM(50, name='lstm_2')(x)
x = Dense(1, name='dense_0')(x)
output = Activation('linear', name='linear_output')(x)

model = Model(inputs=lstm_input, outputs=output)
adam = optimizers.Adam(lr=0.0005)
model.compile(optimizer=adam, loss='mse')

BATCH_SIZE = 64

history = model.fit(
    X_train, 
    y_train, 
    epochs=50, 
    batch_size=BATCH_SIZE, 
    shuffle=True,
    validation_split=0.1
)

model.evaluate(X_test, y_test)

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

y_hat = model.predict(X_test)

y_test_inverse = scaler.inverse_transform(y_test)
y_hat_inverse = scaler.inverse_transform(y_hat)
 
plt.plot(y_test_inverse, label="Actual Price", color='green')
plt.plot(y_hat_inverse, label="Predicted Price", color='red')
 
plt.title('Bitcoin price prediction')
plt.xlabel('Time [days]')
plt.ylabel('Price')
plt.legend(loc='best')
 
plt.show()