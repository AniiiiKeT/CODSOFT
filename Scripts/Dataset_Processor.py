import pandas as pd
import numpy as np

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
 
import regex as re

class DatasetProcessor:
    def __init__(self) -> None:
        self.data = []

    def load_dataset(self, file_path: str, columns = list, save = False) -> pd.DataFrame:
        if file_path.endswith('.txt'):
            if columns is not None:
                with open(file_path, 'r', encoding='utf-8') as file:
                    for line in file:
                        line = line.split(' ::: ')
                        self.data.append(line)
                dataframe = pd.DataFrame(self.data, columns= columns )

                return dataframe
            else:
                raise ValueError("Please provide relevant Column Names")
        
        elif file_path.endswith('.csv'):
            dataframe = pd.read_csv(file_path)
            return dataframe
        else:
            raise ValueError("Unsupported file format. Please use .txt or .csv files.")
    
    def clean_text(self, text: str, *args) -> str:
        '''This method is used to preprocess the textual content  such as getting rid
        of stopwords,tokenization'''
        if len(args) == 0:
            if not isinstance(text, str):
                text = str(text) 
            stop_words = set(stopwords.words('english'))

            def remove_numbers(text: str) -> str:
                return re.sub(r'\d+','', text)
            
            def preprocess(text: str) -> str:
                words = word_tokenize(text.lower())
                stemmer = PorterStemmer()
                filtered_words = [stemmer.stem(word) for word in words if word.isalpha() and word not in stop_words]
                return ' '.join(filtered_words)

            text = remove_numbers(text)
            text = preprocess(text)
            return text
        else:
            texts.append(preprocess(remove_numbers(text)))
            texts = []
            for element in args:
                texts.append(preprocess(remove_numbers(element)))
            return texts

if __name__ == '__main__':
    processor = DatasetProcessor()
    data = processor.load_dataset('Data/Movie_Genre_Classification_Dataset/data.txt', ['ID', 'Title', 'Genre', 'Description'])
    print('BEFORE')
    print(data.head())
    data['Title'] = data['Title'].apply(processor.clean_text)
    print('AFTER')
    print(data.head())