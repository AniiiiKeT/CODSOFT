from Dataset_Processor import DatasetProcessor

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder 
from sklearn.metrics import accuracy_score, classification_report

def prepare_data(file_path : str, ):
    processor = DatasetProcessor()
    data = processor.load_dataset(file_path)
    
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    return x_train,x_test,y_train,y_test
if __name__ == '__main':
    prepare_data(file_path = 'Data\\Credit_Card_Fraud_Detection_Dataset\\fraudTrain.csv', )
    