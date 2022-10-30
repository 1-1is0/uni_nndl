import pandas as pd
import numpy as np
data=pd.read_csv("./MadaLine.csv")

X1=data.iloc[:,0]
X2=data.iloc[:,1]
target=data.iloc[:,2]
class1=data.loc[data["0.0"]==0,:]
class2=data.loc[data["0.0"]==1,:]
