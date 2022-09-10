
import pickle
import numpy as np

class hpp():

    def __init__(self,data) :

        self.data=data

    def predict(self):

        with open('artifacts/model.pkl','rb') as file:
            model=pickle.load(file)    
        CRIM, ZN, INDUS, CHAS, NOX, RM, AGE, DIS, RAD, TAX, PTRATIO, B, LSTAT=self.data.values() 

        array=np.array([CRIM, ZN, INDUS, CHAS, NOX, RM, AGE, DIS, RAD, TAX, PTRATIO, B, LSTAT],ndmin=2,dtype=float)   

        return model.predict(array)  

