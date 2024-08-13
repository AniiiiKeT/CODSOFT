#Custom Imports
from Dataset_Creation import Dataset_Creation

#Library Imports
import pandas as pd 
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB

#Instantiating the data object and getting the dataset
data_object = Dataset_Creation()
data = data_object.dataset_creator('Dataset/Genre Classification Dataset/train_data.txt')
data = data_object.dataset_cleaning(data)

#Initiating the model
model = MultinomialNB()
tfid_vectorizer = TfidfVectorizer()
x = data['DESCRIPTION']
y = data['GENRE']

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=42)
x_train = tfid_vectorizer.fit_transform(x_train)
x_test = tfid_vectorizer.transform(x_test)

model.fit(x_train,y_train)
y_pred = model.predict(x_test)
print(y_pred)