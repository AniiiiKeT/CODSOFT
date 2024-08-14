# Custom Imports
from Dataset_Processor import DatasetProcessor

# Library Imports
import pandas as pd 
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import joblib


# Instantiating the dataset processor object and loading the dataset
# processor = DatasetProcessor()
# data = processor.load_dataset('Dataset/Genre Classification Dataset/train_data.txt')
# cleaned_data = processor.clean_dataset(data)

# cleaned_data.to_csv('Processed_Data.csv')

# Setting up the model
cleaned_data = pd.read_csv('Processed_Data.csv')

# model = MultinomialNB()
model = RandomForestClassifier()
tfidf_vectorizer = TfidfVectorizer()

# Splitting the dataset into features and labels
X = cleaned_data['Description']
y = cleaned_data['Genre']

# Splitting the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Transforming the text data into TF-IDF features
X_train = tfidf_vectorizer.fit_transform(X_train)
X_test = tfidf_vectorizer.transform(X_test)


# Training the model
# model.fit(X_train, y_train)
# joblib.dump(model,'Movie_Genre_Classification_RandomForest.pkl')

model = joblib.load('Movie_Genre_Classification_RandomForest.pkl')
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test,y_pred)
confusion_matrix = confusion_matrix(y_test,y_pred)

print(accuracy)
print(classification_report(y_test,y_pred))



