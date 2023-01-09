import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer


class Process:
    def __init__(self, file_path='data/adult.csv'):
        self.file_path = file_path

    def handle_missing_data(self, df):
        """Handle missing data in the dataset by replacing missing values with the most frequent value.

        Args:
            df (pandas.DataFrame): The dataset.

        Returns:
            pandas.DataFrame: The dataset with missing data handled.
        """
        df.replace('?', np.nan, inplace=True)
        if df.isnull().values.any():
            imputer = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
            df = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)
        return df

    def encode_independent_variables(self, X):
        """Encode the independent variables.

        Args:
            X (numpy.ndarray): The independent variables.

        Returns:
            numpy.ndarray: The encoded independent variables.
        """
        le = LabelEncoder()
        for i in range(X.shape[1]):
            X[:, i] = le.fit_transform(X[:, i])
        return X
        
    def encode_dependent_variable(self, y):
        """Encode the dependent variable.

        Args:
            y (numpy.ndarray): The dependent variable.

        Returns:
            numpy.ndarray: The encoded dependent variable.
        """
        for i in range(len(y)):
            if y[i] == '<=50K':
                y[i] = -1.0
            else:
                y[i] = 1.0
        return y
    
    def fragment_data(self, df):
        """Fragment the dataset into 10 subsets of different sizes.

        Args:
            df (pandas.DataFrame): The original dataset.

        Returns:
            list: The subsets of the dataset.
        """
        subsets = []
        for i in range(10):
            subsets.append(df.sample(frac=0.01 + 0.01 * i, random_state=42))
        return subsets
    
    def load_data(self):
        """Load the dataset.

        Returns:
            list: The subsets of the dataset.
        """
        df = pd.read_csv(self.file_path)
        subsets = self.fragment_data(df)
        for i in range(len(subsets)):
            subsets[i] = self.handle_missing_data(subsets[i])
            subsets[i] = subsets[i].values
            X = subsets[i][:, :-1]
            y = subsets[i][:, -1]
            X = self.encode_independent_variables(X)
            X = StandardScaler().fit_transform(X)
            y = self.encode_dependent_variable(y).astype('int')
            subsets[i] = (X, y)
        return subsets
