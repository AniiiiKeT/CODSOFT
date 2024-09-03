from Dataset_Processor import DatasetProcessor

import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import joblib
import Dataset_Processor

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier 
from sklearn.naive_bayes import MultinomialNB 
from sklearn.metrics import accuracy_score

def prepare_data(file_path: str, columns: list):
    processor = DatasetProcessor()
    data = processor.load_dataset(file_path = file_path, columns = columns)
    x = data[['Title','Description']]
    y = data['Genre']
    x = x.apply(axis = 1, func = processor.clean_text)
    le = LabelEncoder()
    tfidf_vectorizer = TfidfVectorizer()
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    x_train = tfidf_vectorizer.fit_transform(x_train)
    x_test = tfidf_vectorizer.transform(x_test)
    y_train = le.fit_transform(y_train)
    y_test = le.transform(y_test)
    return x_train,x_test,y_train,y_test

def mnb_model(x, y, model_path=None):
    if model_path is None:
        mnbclassifier = MultinomialNB()
        mnbclassifier.fit(x, y)
        joblib.dump(mnbclassifier, 'Model_Assets/Movie_Genre_Classification_MNB.pkl')
        return mnbclassifier
    else:
        model = joblib.load(model_path)
        return model

def rf_model(x, y, model_path=None):
    if model_path == None:
        rfclassifier = RandomForestClassifier()
        rfclassifier.fit(x, y)
        joblib.dump(rfclassifier,'Model_Assets/Movie_Genre_Classification_RF.pkl')
        return rfclassifier
    else:
        model = joblib.load(model_path)
        return model
    
def lg_model(x, y, model_path=None):
    if model_path == None:
        regressor = LogisticRegression(max_iter=1000)
        regressor.fit(x, y)
        joblib.dump(regressor,'Model_Assets/Movie_Genre_Classification_LG.pkl')
        return regressor
    else:
        model = joblib.load(model_path)
        return model


if __name__ == "__main__":

    data_path = 'Data/Movie_Genre_Classification_Dataset/data.txt'
    x_train,x_test,y_train,y_test = prepare_data(file_path = data_path, columns= ['ID', 'Title', 'Genre', 'Description'])

    model1 = mnb_model(x_train,y_train)
    model1_pred = model1.predict(x_test)
    accuracy1 = accuracy_score(y_test, model1_pred)
    print('The accuracy of the Multinomial Naive Bayes model: ',accuracy1)

    model2 = rf_model(x_train,y_train)
    model2_pred = model2.predict(x_test)
    accuracy2 = accuracy_score(y_test, model2_pred)
    print('The accuracy of the Random Forest model: ',accuracy2)

    model3 = lg_model(x_train,y_train)
    model3_pred = model3.predict(x_test)
    accuracy3 = accuracy_score(y_test, model3_pred)
    print('The accuracy of the Logistic Regression model: ',accuracy3)

    

