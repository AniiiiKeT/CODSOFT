import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import regex as re

class DatasetProcessor:
    def __init__(self) -> None:
        self.data = []

    def load_dataset(self, file_path: str) -> pd.DataFrame:
        if file_path.endswith('.txt'):
            with open(file_path, 'r', encoding='utf-8') as file:
                for line in file:
                    line = line.split(' ::: ')
                    self.data.append(line)
            dataframe = pd.DataFrame(self.data, columns=['ID', 'Title', 'Genre', 'Description'])
            return dataframe
        
        elif file_path.endswith('.csv'):
            dataframe = pd.read_csv(file_path)
            return dataframe
        else:
            raise ValueError("Unsupported file format. Please use .txt or .csv files.")
    
    def clean_dataset(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        '''This method is used to preprocess the textual content of the dataset such as getting rid
        of stopwords,tokenization'''
        stop_words = set(stopwords.words('english'))

        def clean_text(text: str) -> str:
            words = word_tokenize(text.lower())
            filtered_words = [word for word in words if word.isalpha() and word not in stop_words]
            return ' '.join(filtered_words)

        def remove_numbers(text: str) -> str:
            return re.sub(r'\d+', '', text)
        
        dataframe['Description'] = dataframe['Description'].astype(str).apply(clean_text)
        dataframe['Description'] = dataframe['Description'].apply(remove_numbers) 
        
        return dataframe

if __name__ == '__main__':
    processor = DatasetProcessor()
    data = processor.load_dataset('Dataset/Genre Classification Dataset/train_data.txt')
    cleaned_data = processor.clean_dataset(data)
    print(cleaned_data.head())
