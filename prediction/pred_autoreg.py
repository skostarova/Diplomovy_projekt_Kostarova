import numpy as np
import pandas as pd
import keras
import os
    
    
def run():

    f_name = "shift-15-windows-45"
    m_names = ["epoch-09-mae-0.0318.hdf5"] #os.listdir(f'../newModel/autoreg/{f_name}/') 
    part = 1
    
    X_test = np.load(f'../data/autoreg/{f_name}/X_test_min.npy', allow_pickle=True)
    y_test = np.load(f'../data/autoreg/{f_name}/y_test.npy', allow_pickle=True)    
    
    Phi60_Sig1 = X_test["Phi60_Sig1"]

    y_test = y_test[ :int(len(y_test)/part)]
    Phi60_Sig1 = Phi60_Sig1[ :int(len(Phi60_Sig1)/part)]
###############################################################################
    os.makedirs(f"autoreg/{f_name}", exist_ok=True)

# predict
    for saved_model in m_names:
        print(saved_model)

        model = keras.models.load_model(f"../newModel/autoreg/{f_name}/{saved_model}")
        y_pred = model.predict(Phi60_Sig1, verbose = 1)
        model_name = saved_model[ :(len(saved_model)-5)]
        np.save(f'autoreg/{f_name}/y_pred_{model_name}.npy', y_pred)
	
        
        
if __name__ == '__main__':
    run()
    