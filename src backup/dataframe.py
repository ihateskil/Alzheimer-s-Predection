import pandas as pd
from sklearn.model_selection import train_test_split


# loading the dataset
data_frame = pd.read_csv('C:/Users/iiski/Desktop/Alzheimer_Prediction_Project/datasets/alzheimers_disease_data.csv')
# Data separation into input data and result
input_data = data_frame.drop(columns=['PatientID', 'Diagnosis'])
result = data_frame['Diagnosis']
# Splitting the dataset into 80% training and 20% testing
input_data_train, input_data_test, result_train, result_test = train_test_split(input_data, result, test_size=0.2, random_state=17)

