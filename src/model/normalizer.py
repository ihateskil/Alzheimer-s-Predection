import pandas as pd
from sklearn.preprocessing import MinMaxScaler

class Normalizer:
    def __init__(self):
        self.scaler = MinMaxScaler()
        self.columns = None
    
    def fit(self, data):
        self.columns = data.columns
        self.scaler.fit(data)
    
    def transform(self, data):
        # Check if the Normalizer is fitted
        if self.columns is None:
            raise ValueError("Normalizer must be fitted before transforming.")
        # Check if the MinMaxScaler is fitted by checking its attributes
        if not hasattr(self.scaler, 'data_min_') or self.scaler.data_min_ is None:
            raise ValueError("MinMaxScaler must be fitted before transforming.")
        if not all(col in data.columns for col in self.columns):
            raise ValueError("Input data must have the same columns as the fitted data.")
        
        normalized_data = self.scaler.transform(data)
        return pd.DataFrame(
            normalized_data,
            columns=self.columns,
            index=data.index
        )
    
    def fit_transform(self, data):
        self.fit(data)
        return self.transform(data)