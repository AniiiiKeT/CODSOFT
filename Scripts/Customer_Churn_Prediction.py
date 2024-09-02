import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import joblib

from Dataset_Processor import DatasetProcessor

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier 
from sklearn.preprocessing import LabelEncoder, TargetEncoder
from sklearn.metrics import accuracy_score, classification_report

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Softmax

def prepare_data(file_path : str):
    processor = DatasetProcessor()
    le = LabelEncoder()
    se = StandardScaler()
    data = processor.load_dataset("D:/CODING/CODSOFT/Data/Churn_Prediction_Dataset/Churn_Modelling.csv")
    data.drop(['CustomerId','RowNumber','Surname'], axis = 1, inplace = True)

    categorical_data = ['Geography','Gender']
    numerical_data = [column for column in data.columns[0:-1] if column not in categorical_data]
    x_train,x_test,y_train,y_test = train_test_split(data.iloc[:,0:-1],data.iloc[:,-1], stratify = data.iloc[:,-1], test_size = 0.2, random_state = 42)

    for column in categorical_data:
        x_train[column] = le.fit_transform(x_train[column])
        x_test[column] = le.transform(x_test[column])
        
    x_train[numerical_data] = se.fit_transform(pd.DataFrame(x_train[numerical_data],columns = numerical_data))
    x_test[numerical_data] = se.transform(pd.DataFrame(x_test[numerical_data],columns = numerical_data))

    return x_train,x_test,y_train,y_test

def visualize() -> None:
    processor = DatasetProcessor()
    data = processor.load_dataset("D:/CODING/CODSOFT/Data/Churn_Prediction_Dataset/Churn_Modelling.csv")
    summed_data = data[['Geography','Exited']].groupby('Geography').count()
    summed_data2 = data[['Gender','Exited']].groupby('Gender').count()
    figure, axes = plt.subplots(2,figsize=(6, 6))
    axes[0].bar(height = summed_data2['Exited'], x= summed_data2.index)
    axes[0].set_title('Number of People Exited according to sex')
    axes[1].pie(summed_data['Exited'],labels = summed_data.index)
    axes[1].set_title('Number of customers Exited per country')
    plt.tight_layout()
    plt.show()

def random_forest_model(x,y,model_path = None):
    if model_path == None: 
        model = RandomForestClassifier()
        model.fit(x,y)
        joblib.dump(model,'D:/CODING/CODSOFT/Model_Assets/CCP_RF.pkl')
        return model
    else:
        model = joblib.load(model_path)
        return model

def lg_model(x, y, model_path=None):
    if model_path == None:
        regressor = LogisticRegression(max_iter=1000)
        regressor.fit(x, y)
        joblib.dump(regressor,'D:/CODING/CODSOFT/Model_Assets/CCP_LG.pkl')
        return regressor
    else:
        model = joblib.load(model_path)
        return model

def knb_model(x, y, model_path = None):
    if model_path == None:
        model = KNeighborsClassifier(n_neighbors = 10)
        model.fit(x, y)
        joblib.dump(model,'D:/CODING/CODSOFT/Model_Assets/CCP_KNB.pkl')
        return model
    else:
        model = joblib.load(model_path)
        return model

def dl_model(x, y, model_path = None):
    if model_path == None:
        model = Sequential([
            Dense(128,input_shape = (10,),activation = 'relu'),
            Dropout(0.1),
            Dense(64,activation = 'relu'),
            Dropout(0.1),
            Dense(32,activation = 'relu'),
            Dropout(0.1),
            Dense(1,activation = 'sigmoid'),
        ])
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        model.fit(x, y, epochs = 10)
        model.save('D:/CODING/CODSOFT/Model_Assets/CCP_DL.h5', model)
        return model
    else:
        model = load_model(model_path)
        return model

if __name__ == '__main__':
    x_train,x_test,y_train,y_test = prepare_data("Data/Churn_Prediction_Dataset/Churn_Modelling.csv")
    visualize()

    random_forest = random_forest_model(x_train,y_train)
    logistic_regression = lg_model(x_train,y_train)
    knb_model = knb_model(x_train,y_train)

    y_pred1 = random_forest.predict(x_test)
    y_pred2 = logistic_regression.predict(x_test)
    y_pred3 = knb_model.predict(x_test)

    accuracy1 = accuracy_score(y_test,y_pred1)
    accuracy2 = accuracy_score(y_test,y_pred2)
    accuracy3 = accuracy_score(y_test,y_pred3)

    print(
        {
        'Random Forest Model Accuracy' : accuracy1,
        'Logistic Regression Model Accuracy' : accuracy2,
        'K Nearest Neighbours Model Accuracy' : accuracy3

        })
