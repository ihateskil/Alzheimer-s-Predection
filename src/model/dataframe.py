import pandas as pd
from sklearn.model_selection import train_test_split

class DataFrame:
    def __init__(self):
        # Load the dataset
        self.data_frame = pd.read_csv('C:/Users/iiski/Desktop/Alzheimer_Prediction_Project/datasets/alzheimers_disease_data.csv')
        self.input_data = self.data_frame.drop(columns=['PatientID', 'Diagnosis'])
        self.result = self.data_frame['Diagnosis']

        # Split the dataset into 80% training and 20% testing
        self.input_data_train, self.input_data_test, self.result_train, self.result_test = train_test_split(
            self.input_data, self.result, test_size=0.2, random_state=17
        )

    def get_data(self):
        return self.input_data_train, self.input_data_test, self.result_train, self.result_test