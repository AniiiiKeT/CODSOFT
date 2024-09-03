import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import joblib

from Dataset_Processor import DatasetProcessor

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier 
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report

import tensorflow 
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Dense, Dropout, Input

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

    return x_train, x_test, y_train, y_test, tfidf

def classify_input(text : str, object):
    processor = DatasetProcessor()
    text = processor.clean_text(text)
    text = object.transform([text])
    text = text.toarray()
    return text

def random_forest_model(x, y, model_path = None):
    if model_path == None:
        model = RandomForestClassifier()
        model.fit(x,y)
        joblib.dump(model, 'Model_Assets/SH_RF.pkl')
        return model
    else:
        model = joblib.load(model_path)
        return model

def dl_model(x ,y, model_path = None):
    if model_path is None:
        model = Sequential([
            Input(shape = (x_train.shape[1],)),
            Dense(128, activation = 'relu'),
            Dropout(0.25),
            Dense(64, activation = 'relu'),
            Dropout(0.25),
            Dense(32, activation = 'relu'),
            Dropout(0.25),
            Dense(1, activation = 'sigmoid'),
        ])
        model.compile(optimizer = 'adam', metrics = ['accuracy'], loss = 'binary_crossentropy')
        model.fit(x, y, epochs = 10)
        model.save('Model_Assets/SH_DL.h5')
        return model
    else:
        model = load_model(model_path)
        return model

def mnb_model(x, y, model_path=None):
    if model_path is None:
        mnbclassifier = MultinomialNB()
        mnbclassifier.fit(x, y)
        joblib.dump(mnbclassifier, 'Model_Assets/SH_MNB.pkl')
        return mnbclassifier
    else:
        model = joblib.load(model_path)
        return model
    
def lg_model(x, y, model_path=None):
    if model_path == None:
        regressor = LogisticRegression(max_iter=1000)
        regressor.fit(x, y)
        joblib.dump(regressor,'Model_Assets/SH_LG.pkl')
        return regressor
    else:
        model = joblib.load(model_path)
        return model

if __name__ == '__main__':
    x_train, x_test, y_train, y_test, tfidf = prepare_data("Data/Spam_Ham_Classification_Dataset/spam.csv")

     # Train and evaluate Random Forest
    random_forest = random_forest_model(x_train, y_train)
    y_pred_rf = random_forest.predict(x_test)
    accuracy_rf = accuracy_score(y_test, y_pred_rf)
    print(f'Random Forest Accuracy: {accuracy_rf}')
    print(f'Random Forest Classification Report:\n{classification_report(y_test, y_pred_rf)}')

    # Train and evaluate Deep Learning model
    deep_learning_model = dl_model(x_train, y_train)
    y_pred_dl = deep_learning_model.predict(x_test)
    y_pred_dl = [1 if pred >= 0.5 else 0 for pred in y_pred_dl]
    accuracy_dl = accuracy_score(y_test, y_pred_dl)
    print(f'Deep Learning Model Accuracy: {accuracy_dl}')
    print(f'Deep Learning Classification Report:\n{classification_report(y_test, y_pred_dl)}')

    # Train and evaluate Multinomial Naive Bayes
    mnb = mnb_model(x_train, y_train)
    y_pred_mnb = mnb.predict(x_test)
    accuracy_mnb = accuracy_score(y_test, y_pred_mnb)
    print(f'Multinomial Naive Bayes Accuracy: {accuracy_mnb}')
    print(f'Multinomial Naive Bayes Classification Report:\n{classification_report(y_test, y_pred_mnb)}')

    # Train and evaluate Logistic Regression
    logistic_regression = lg_model(x_train, y_train)
    y_pred_lg = logistic_regression.predict(x_test)
    accuracy_lg = accuracy_score(y_test, y_pred_lg)
    print(f'Logistic Regression Accuracy: {accuracy_lg}')
    print(f'Logistic Regression Classification Report:\n{classification_report(y_test, y_pred_lg)}')

    # Classify user input text
    input_text = input('Enter the text you want to classify: ')
    text_vectorized = classify_input(input_text, tfidf)
    y_pred_user_dl = deep_learning_model.predict(text_vectorized)
    result_dl = 'spam' if y_pred_user_dl[0] >= 0.5 else 'ham'
    print(f'\nPrediction from Deep Learning Model: {result_dl}')

    y_pred_user_mnb = mnb.predict(text_vectorized)
    result_mnb = 'spam' if y_pred_user_mnb[0] == 1 else 'ham'
    print(f'Prediction from Multinomial Naive Bayes: {result_mnb}')

    y_pred_user_lg = logistic_regression.predict(text_vectorized)
    result_lg = 'spam' if y_pred_user_lg[0] == 1 else 'ham'
    print(f'Prediction from Logistic Regression: {result_lg}')

    y_pred_user_rf = random_forest.predict(text_vectorized)
    result_rf = 'spam' if y_pred_user_rf[0] == 1 else 'ham'
    print(f'Prediction from Random Forest: {result_rf}')