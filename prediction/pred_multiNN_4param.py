import numpy as np
import pandas as pd
import keras
import os
    
    
def run():
    
    f_name = "shift-15-windows-45"
    param = "all_parameters"
    m_names = ["epoch-02-mae-0.0134.hdf5"] 
    #m_names = os.listdir(f"../newModel/{param}/{f_name}/") 	# for all epochs
    size=1
    X_test = np.load(f'../data/{param}/{f_name}/X_test_min.npy', allow_pickle=True)
    X_test_hour = np.load(f'../data/{param}/{f_name}/X_test_hour.npy', allow_pickle=True)
    y_test = np.load(f'../data/{param}/{f_name}/y_test.npy', allow_pickle=True)    
    

    Phi60_Sig1 = X_test["Phi60_Sig1"]

    AsyH = X_test["AsyH"]
    ap_index = X_test_hour["ap_index"]
    PC = X_test["PC"]
    BzGSE = X_test["BzGSE"]

    y_test = y_test[ :int(len(y_test)/size)]

    Phi60_Sig1 = Phi60_Sig1[ :int(len(Phi60_Sig1)/size)]
   


    AsyH = AsyH[ :int(len(AsyH)/size)]
    ap_index = ap_index[ :int(len(ap_index)/size)]
    PC = PC[ :int(len(PC)/size)]
    BzGSE = BzGSE[ :int(len(BzGSE)/size)]

###############################################################################
    os.makedirs(f"{param}/{f_name}", exist_ok=True)
# predict
    for saved_model in m_names:
        print(saved_model)
        
        model = keras.models.load_model(f"../newModel/{param}/{f_name}/{saved_model}")
        y_pred = model.predict([ap_index, BzGSE, AsyH, PC, Phi60_Sig1])
        model_name = saved_model[ :(len(saved_model)-5)]
        np.save(f'{param}/{f_name}/y_pred_{model_name}.npy', y_pred)

        
if __name__ == '__main__':
    run()
    