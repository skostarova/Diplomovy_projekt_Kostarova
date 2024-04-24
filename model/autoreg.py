from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Input, Conv1D, LSTM, MaxPooling1D, Flatten, concatenate, TimeDistributed, Bidirectional, ConvLSTM2D
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.models import Model
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, mean_absolute_error
import os
import numpy as np

model_name = "autoreg"
f_name = f"data/{model_name}"
f_data = "shift-15-windows-45"
X_train = np.load(f'{f_name}/{f_data}/X_train_min.npy', allow_pickle=True)
y_train = np.load(f'{f_name}/{f_data}/y_train.npy', allow_pickle=True)

Phi60_Sig1 = X_train["Phi60_Sig1"]

inputs_Phi60_Sig1 = Input(shape=(Phi60_Sig1.shape[1],1))
a = Bidirectional(LSTM(128, dropout=0.1,recurrent_dropout=0.1))(inputs_Phi60_Sig1)
a = Dense(64, activation='relu')(a)
a = Dropout(0.2)(a)
a = Dense(64, activation='relu')(a)
a = Dense(32, activation='relu')(a)
output = Dense(1, activation='linear')(a)

model = Model(inputs=inputs_Phi60_Sig1, outputs=output)
model.compile(loss='mse', optimizer='adam', metrics=["mae"])
print(model.summary())

# callbacks
model_f_name = f"newModel/{model_name}/{f_data}/"
os.makedirs(model_f_name, exist_ok=True)
saved_model = "/epoch-{epoch:02d}-mae-{val_mae:.4f}.hdf5"
checkpoint = ModelCheckpoint(model_f_name + saved_model, monitor='val_mae', verbose=1, save_freq="epoch")
callbacks_list = [checkpoint]

history = model.fit(
    x=Phi60_Sig1,
    y=y_train,
    validation_split=0.1,
    epochs=10,
    verbose=1,
    callbacks=callbacks_list,
    shuffle=False
)