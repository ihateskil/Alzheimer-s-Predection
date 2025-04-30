import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, mutual_info_classif

class DataFrame:
    def __init__(self):
        

        # loading the dataset
        self.data_frame = pd.read_csv('C:/Users/iiski/Desktop/Alzheimer_Prediction_Project/datasets/alzheimers_disease_data.csv')
        # Data separation into input data and result
        self.input_data = self.data_frame.drop(columns=['PatientID', 'Diagnosis'])
        self.result = self.data_frame['Diagnosis']
        # Splitting the dataset into 80% training and 20% testing
        self.input_data_train, self.input_data_test, self.result_train, self.result_test = train_test_split(self.input_data, self.result, test_size=0.2, random_state=17)

    def __call__(self):
        # Select the top 10 most valuable features using mutual information

        # Initialize SelectKBest with mutual information
        selector = SelectKBest(score_func=mutual_info_classif, k=10)
        
        # Fit selector on training data and transform both train and test
        selected_train = selector.fit_transform(self.input_data_train, self.result_train)
        selected_test = selector.transform(self.input_data_test)
        
        # Get the indices of the selected features
        selected_indices = selector.get_support(indices=True)
        selected_columns = self.input_data_train.columns[selected_indices].tolist()
        
        # Convert back to DataFrame with selected columns
        self.input_data_train_selected = pd.DataFrame(
            selected_train,
            columns=selected_columns,
            index=self.input_data_train.index
        )
        self.input_data_test_selected = pd.DataFrame(
            selected_test,
            columns=selected_columns,
            index=self.input_data_test.index
        )
        
        # Return the selected training and testing data
        return self.input_data_train_selected, self.input_data_test_selected, self.result_train, self.result_test