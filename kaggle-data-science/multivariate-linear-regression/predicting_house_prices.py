import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Making sure that the class variable is on the -1 index and not in the middle of the dataset
dataset = pd.read_csv('kc_house_data.csv')
price = dataset.pop('price')
sqft_lot15 = dataset.pop('sqft_lot15')
dataset['sqft_lot15'] = sqft_lot15
dataset['price'] = price

# Removing the 'T000000' present at the end of the strings in the original format
def strip_string(string):
    string = string[:8]
    return string

dataset['date'] = dataset['date'].apply(strip_string)
dataset['date'] = pd.to_datetime(dataset['date'], format='%Y%m%d')

# for every column, get the vector of rows inside the whisker's range
# afterwards, use set operations to modify the dataset in such a way that it only has the indexes that haven't been picked out

def remove_outliers(dataset):
    dataset_columns = list(dataset.columns)
    for column in dataset_columns:
        vector = dataset[column]
        q25 = vector.quantile(0.25)
        q75 = vector.quantile(0.75)
        IQR = q75 - q25
        vector = vector[(vector >= q25 - IQR*1.5) & (vector <= q75 + IQR*1.5)]
        dataset = dataset.loc[list(set(dataset.index).intersection(set(vector.index)))] 
    return dataset

dataset = remove_outliers(dataset.drop(['id', 'date', 'zipcode'], 1))

X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)