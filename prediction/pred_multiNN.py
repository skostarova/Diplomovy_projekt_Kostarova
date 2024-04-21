import numpy as np
import pandas as pd
import keras
import os

    
def run():

    f_name = "shift-15-windows-45"
    param = "PC"
    m_names = ['epoch-06-mae-0.0284.hdf5','epoch-09-mae-0.0287.hdf5'] #for all epochs - os.listdir(f'../newModel/{param}/{f_name}/')
    part=1 		# choose 1/part subset from test_data 
    
    X_test = np.load(f'../data/{param}/{f_name}/X_test_min.npy', allow_pickle=True)
    #X_test_hour = np.load(f'../data/{param}/{f_name}/X_test_hour.npy', allow_pickle=True) # for ap_index
    y_test = np.load(f'../data/{param}/{f_name}/y_test.npy', allow_pickle=True)    
    

    Phi60_Sig1 = X_test["Phi60_Sig1"]
    X_test_param = X_test[param]     # for ap_index - X_test_hour[param]

    y_test = y_test[ :int(len(y_test)/part)]

    Phi60_Sig1 = Phi60_Sig1[ :int(len(Phi60_Sig1)/part)]
    X_test_param = X_test_param[ :int(len(X_test_param)/part)]


###############################################################################
    os.makedirs(f"{param}/{f_name}", exist_ok=True)
# predict
    for saved_model in m_names:
        print(saved_model)
        
        model = keras.models.load_model(f"../newModel/{param}/{f_name}/{saved_model}")
        y_pred = model.predict([X_test_param, Phi60_Sig1])
        model_name = saved_model[ :(len(saved_model)-5)]
        np.save(f"{param}/{f_name}/y_pred_{model_name}.npy", y_pred)
        

        
if __name__ == '__main__':
    run()
    