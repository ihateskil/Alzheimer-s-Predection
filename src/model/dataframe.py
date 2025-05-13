import pandas as pd
from sklearn.model_selection import train_test_split

class DataFrame:
    def __init__(self):
        self._load_and_prepare_data()

    def _load_and_prepare_data(self):
        data_path = 'Alzheimer_Prediction_Project/datasets/alzheimers_disease_data.csv' # ADD YOUR OWN PATH HERE
        self.data_frame = pd.read_csv(data_path)
        self.input_data = self.data_frame.drop(columns=['PatientID', 'Diagnosis'])
        self.result = self.data_frame['Diagnosis']
        self._split_data()

    def _split_data(self):
        self.input_data_train, self.input_data_test, self.result_train, self.result_test = train_test_split(
            self.input_data, self.result, test_size=0.2, random_state=17
        )

    def get_data(self):
        return self.input_data_train, self.input_data_test, self.result_train, self.result_test
