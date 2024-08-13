import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import regex as re

#Run this lines on the first execution
# nltk.download('stopwords')
# nltk.download('punkt')

class Dataset_Creation:
    dataset = []
    def __init__(self) -> None:
        pass

    def dataset_creator(self, directory):
        if directory.endswith('.txt'):
            with open(directory,'r',encoding='utf-8') as file:
                for line in file:
                    line = line.split(' ::: ')
                    self.dataset.append(line)
            data = pd.DataFrame(self.dataset, columns=['ID','TITLE','GENRE','DESCRIPTION'])
            return data
        
        elif directory.endswith('.csv'):
            data = pd.read_csv(directory)
            return data
    
    def dataset_cleaning(self, data):
        stop_words = set(stopwords.words('english'))

        def clean_text(text):
            words = word_tokenize(text.lower())
            filtered_words = [word for word in words if word.isalpha() and word not in stop_words]
            return ' '.join(filtered_words)
        def remove_numbers(text):
            return re.sub(r'\d+', '', text)
        
        data['DESCRIPTION'] = data['DESCRIPTION'].astype(str).apply(clean_text)
        data['DESCRIPTION'] = data['DESCRIPTION'].apply(remove_numbers) 
        
        return data

if __name__ == '__main__':
    dataset_creator = Dataset_Creation()
    data = dataset_creator.dataset_creator('Dataset/Genre Classification Dataset/train_data.txt')
    data = dataset_creator.dataset_cleaning(data)
    print(data.head())