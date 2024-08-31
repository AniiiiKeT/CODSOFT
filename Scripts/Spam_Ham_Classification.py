import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import joblib

from Dataset_Processor import DatasetProcessor

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier 
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report

import tensorflow 
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Dense, Dropout

def prepare_data(file_path : str):
    processor = DatasetProcessor()
    le = LabelEncoder()
    tfidf = TfidfVectorizer()

    data = processor.load_dataset(file_path = file_path, encoding ='latin1')
    data.drop(['Unnamed: 2','Unnamed: 3','Unnamed: 4'],axis = 1, inplace = True)
    x = data['v2']
    y = data['v1']
    x = x.apply(processor.clean_text)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 42)
    
    x_train = tfidf.fit_transform(x_train)
    x_test = tfidf.transform(x_test)
    y_train = le.fit_transform(y_train)
    y_test = le.transform(y_test)

    return x_train, x_test, y_train, y_test

def random_forest_model(x, y, model_path = None):
    if model_path == None:
        model = RandomForestClassifier()
        model.fit(x,y)
        joblib.dump(model, 'Model_Assets/SH_RF.pkl')
        return model
    else:
        model = joblib.load(model_path)
        return model

if __name__ == '__main__':
    x_train, x_test, y_train, y_test = prepare_data("Data/Spam_Ham_Classification_Dataset/spam.csv")

    random_forest = random_forest_model(x_train,y_train)
    y_pred1 = random_forest.predict(x_test)
    accuracy1 = accuracy_score(y_test,y_pred1)

    print(accuracy1)
