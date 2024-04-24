# FOR multiNN with all_parameters


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

model_name = "all_parameters"
f_data=f"{model_name}/shift-15-windows-45"
f_name=f"newModel/{f_data}/"


#Load data
with open(f'data/{f_data}/X_train_min.npy', 'rb') as f:
    X_min = pickle.load(f)

X_Phi60_Sig1=X_min["Phi60_Sig1"]


X_PC=X_min["PC"]


X_AsyH=X_min["AsyH"]


X_BzGSE=X_min["BzGSE"]



with open(f'data/{f_data}/X_train_hour.npy', 'rb') as f:
    X_hour = pickle.load(f)
X_ap=X_hour["ap_index"]



with open(f'data/{f_data}/y_train.npy', 'rb') as f:
    y_train = pickle.load(f)


print("len", len(X_Phi60_Sig1), len(X_PC), len(X_ap), len(X_BzGSE), len(y_train))

#Model
# Define 5 sets of inputs
input_Phi60 = Input(shape=(X_Phi60_Sig1.shape[1], 1))
input_PC = Input(shape=(X_PC.shape[1], 1))
input_AsyH = Input(shape=(X_AsyH.shape[1], 1))
input_BzGSE = Input(shape=(X_BzGSE.shape[1], 1))
input_ap = Input(shape=(X_ap.shape[1], 1))

# First branch
a = Bidirectional(LSTM(64, dropout=0.1, recurrent_dropout=0.1))(input_Phi60)
a = Dense(32, activation='relu')(a)

# Second branch
b = Bidirectional(LSTM(64, dropout=0.1, recurrent_dropout=0.1))(input_PC)
b = Dense(32, activation='relu')(b)

# Third branch
c = Bidirectional(LSTM(64, dropout=0.1, recurrent_dropout=0.1))(input_AsyH)
c = Dense(32, activation='relu')(c)

# Fourth branch
d = Bidirectional(LSTM(64, dropout=0.1, recurrent_dropout=0.1))(input_BzGSE)
d = Dense(32, activation='relu')(d)

# Fifth branch
e = Bidirectional(LSTM(64, dropout=0.1, recurrent_dropout=0.1))(input_ap)
e = Dense(32, activation='relu')(e)


# Combine the output of the 5 branches
x = concatenate([a, b, c, d, e])
print(x.shape)
x = Dense(32, activation='relu')(x)
x = Dropout(0.2)(x)  # Consider removing this line or try different dropout rates
x = Dense(32, activation='relu')(x)

# Apply a final Dense layer for regression prediction
output = Dense(1, activation='linear')(x)

# Create and compile the model
classifier = Model(inputs=[input_ap, input_BzGSE, input_AsyH, input_PC, input_Phi60], outputs=output)
classifier.compile(loss='mse', optimizer="adam", metrics=['mae'])  
print(classifier.summary())


os.makedirs(f_name, exist_ok=True)
saved_model = "/epoch-{epoch:02d}-mae-{val_mae:.4f}.hdf5"
checkpoint = ModelCheckpoint(f_name+saved_model, monitor='val_mae', verbose=1, save_freq="epoch")
callbacks_list = [checkpoint]

history = classifier.fit(
    x=[X_ap, X_BzGSE, X_AsyH, X_PC, X_Phi60_Sig1],
    y=y_train,
    validation_split=0.1,
    epochs=10,
    verbose=1,
    callbacks=callbacks_list,
    shuffle = True
)