# FOR multiNN only HOURLY FEATURES (ap_index)

# import
from tensorflow import keras
from keras.preprocessing.sequence import TimeseriesGenerator
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Input, Conv1D, LSTM, MaxPooling1D, Flatten, concatenate, TimeDistributed, Bidirectional, ConvLSTM2D
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.models import Model
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix, classification_report, mean_absolute_error
import os
import numpy as np
import pickle

# settings
model_name = "ap_index"
f_data=f"{model_name}/shift-15-windows-45"
f_name=f"newModel/{f_data}/"

#Load data
with open(f'data/{f_data}/X_train_hour.npy', 'rb') as f:
    X_hour = pickle.load(f)
X_hour=X_hour["ap_index"]

with open(f'data/{f_data}/X_train_min.npy', 'rb') as f:
    X_min = pickle.load(f)
X_min=X_min["Phi60_Sig1"]

with open(f'data/{f_data}/y_train.npy', 'rb') as f:
    y_train = pickle.load(f)




#Model
# Define two sets of inputs
input_kp = Input(shape=(X_hour.shape[1], 1))
input_Phi60 = Input(shape=(X_min.shape[1], 1))

# First branch
a = Bidirectional(LSTM(64, dropout=0.1, recurrent_dropout=0.1))(input_Phi60)
a = Dense(32, activation='relu')(a)

# Second branch
b = Bidirectional(LSTM(64, dropout=0.1, recurrent_dropout=0.1))(input_kp)
b = Dense(32, activation='relu')(b)

# Combine the output of the two branches
x = concatenate([a, b])
x = Dense(32, activation='relu')(x)
x = Dropout(0.2)(x)  # Consider removing this line or try different dropout rates
x = Dense(32, activation='relu')(x)

# Apply a final Dense layer for regression prediction
output = Dense(1, activation='linear')(x)

# Create and compile the model
classifier = Model(inputs=[input_kp, input_Phi60], outputs=output)
classifier.compile(loss='mse', optimizer="adam", metrics=['mae'])  
print(classifier.summary())

# callbacks
os.makedirs(f_name, exist_ok=True)
saved_model = "/epoch-{epoch:02d}-mae-{val_mae:.4f}.hdf5"
checkpoint = ModelCheckpoint(f_name+saved_model, monitor='val_mae', verbose=1, save_freq="epoch")
callbacks_list = [checkpoint]

# training
history = classifier.fit(
    x=[X_hour, X_min],
    y=y_train,
    validation_split=0.1,
    epochs=10,
    verbose=1,
    callbacks=callbacks_list,
    shuffle = True
)
