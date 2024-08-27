from Shared.Dataset_Processor import DatasetProcessor
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

processor = DatasetProcessor()
data = processor.load_dataset('Datasets/fraudTrain.csv')
data.drop(['Unnamed: 0','street'], axis= 1 , inplace= True)


# le = LabelEncoder()
# data['job'] = le.fit_transform(data['job'])
# print(data['job'])

#Performing EDA on the jobs and Frauds
unique = []
for job in data['job']:
    if job not in unique:
        unique.append(job)
print(unique)