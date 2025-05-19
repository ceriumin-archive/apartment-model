import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

'''This class contains all the methods for handling data and pre-processing'''
class DataLoader:
    def __init__(self, filepath, target):
        self.target = target
        self.filepath = filepath
        self.data = None
        self.X_train, self.X_test, self.Y_train, self.Y_test = None, None, None, None

    # Uses ISO encoding as the compiler trips up when not using anything to encode the data
    def load_data(self):
        self.data = pd.read_csv(self.filepath, encoding='ISO-8859-1')

    # Function for containing all the pre-processing steps
    def preprocess_data(self):
        # Removes any unnecessary features, and replaces the original dataset 
        self.data.drop(columns=['Unnamed: 0', 'id', 'address'], inplace=True)
        # Although this is unnecessary it is there in-case, or if the dataset updates
        self.data = self.data.dropna()
        # Encodes categories using One-Hot encoding
        self.data = pd.get_dummies(self.data)

        # Skews any values logarithmically to remove outliers, 0.5 seems to be the sweet spot for this
        for column in self.data.select_dtypes(include=[np.number]).columns:
            if abs(self.data[column].skew()) > 0.5:
                self.data[column] = np.log1p(self.data[column])

    # Data splitting function using skikit test split
    def split_data(self):
        Y = self.data[self.target]
        X = self.data.drop(columns=['price'], axis=1)

        X = (X - X.mean()) / (X.std()) # Feature scaling using a Z score

        # Training split of 1/4 seems to yield the best results
        self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    # Getter function for checking the pre-processed data
    def get_data(self):
        return self.X_train, self.X_test, self.Y_train, self.Y_test
