import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

from Dataset_Processor import DatasetProcessor

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier 
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import TargetEncoder

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import load_model

def prepare_data(file_path: str):
    le = LabelEncoder()
    se = StandardScaler()
    te = TargetEncoder(smooth = 'auto') 
    processor = DatasetProcessor()
    data = processor.load_dataset(file_path)
    data['name'] = data['first'] + " " + data['last']
    data.drop(columns = ['first', 'last'], inplace = True)
    
    categorical_data = ['merchant','category','gender','name','street','city','state','job']
    numerical_data = ['amt','merch_lat','merch_long']

    x_train, x_test, y_train, y_test = train_test_split(data[categorical_data + numerical_data], data['is_fraud'], test_size=0.3, random_state=42)
    x_train_cat = te.fit_transform(x_train[categorical_data], y_train)
    x_test_cat = te.transform(x_test[categorical_data])
    x_train_num = se.fit_transform(x_train[numerical_data])
    x_test_num = se.transform(x_test[numerical_data])
    x_train_final = pd.concat([pd.DataFrame(x_train_cat, columns=categorical_data), pd.DataFrame(x_train_num, columns=numerical_data)], axis=1)
    x_test_final = pd.concat([pd.DataFrame(x_test_cat, columns=categorical_data), pd.DataFrame(x_test_num, columns=numerical_data)], axis=1)
        
    return x_train_final, x_test_final, y_train, y_test  

def logistic_regression_model(x, y, model_path = None):
    if model_path == None:
        model = LogisticRegression(multi_class = 'auto')
        model.fit(x,y)
        return model
    else:
        model = joblib.load(model_path)
        return model

def random_forest_model(x, y, model_path = None):
    if model_path == None:
        model = RandomForestClassifier()
        model.fit(x,y)
        return model
    else:
        model = joblib.load(model_path)
        return model
    
def deep_learning_model(x, y, model_path = None):
    if model_path == None:
        model = Sequential([
            Dense(128, input_shape = (x.shape[1],), activation = 'relu'),
            Dense(64, activation = 'relu'),
            Dense(1, activation = 'sigmoid')
        ])
        model.compile(optimizer = 'adam', metrics = ['accuracy'], loss = 'binary_crossentropy')
        model.fit(x_train,y_train, epochs = 5, batch_size = 32)
        return model
    else:
        model = load_model(model_path)
        return model
    
def show_result(y_test,y_pred):
    result = {
        'True Values' : y_test,
        'Predicted Values' : y_pred,
    }
    result = pd.DataFrame(result)
    return result

if __name__ == '__main__':

    x_train, x_test, y_train, y_test = prepare_data(file_path = 'Data/Credit_Card_Fraud_Detection_Dataset/fraudTrain.csv')
    regressor = logistic_regression_model(x_train,y_train, 'Model_Assets/CCFD_Regressor.pkl')
    random_forest_classifier = random_forest_model(x_train,y_train, 'Model_Assets/CCFD_RandomForestCLassifier.pkl')
    deep_learning_classifier = deep_learning_model(x_train,y_train, 'Model_Assets/CCFD_DL.h5')

    y_pred1 = regressor.predict(x_test)

    print(show_result(y_test,y_pred1))
