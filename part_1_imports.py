# import libraries
import numpy as np                                
import pandas as pd                               
import matplotlib.pyplot as plt                   
from IPython.display import Image                 
from IPython.display import display               
from time import gmtime, strftime                

import subprocess
import sys

from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.random import sample_without_replacement
from sklearn.utils import shuffle
from sklearn.model_selection import KFold
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score

import boruta
from boruta import BorutaPy

#Expanding table width:
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

#Ignore warnings:
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
print('x' in np.arange(5))   #returns False, without Warning

print('imports complete') 

#Installs:
def install(lime):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

def install(boruta):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])
    
print('installs complete')      
