import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from src.model.normalizer import Normalizer
from src.model.dataframe import DataFrame

class AlzheimerModel:
    def __init__(self):
        self.model = SVC(kernel='linear', probability=True, random_state=42)
        self.normalizer = Normalizer()
        self.selector = SelectKBest(score_func=mutual_info_classif, k=10)
        self.feature_columns = None
        self.selected_columns = None

        # Load data using DataFrame
        df_processor = DataFrame()
        self.input_data_train, self.input_data_test, self.result_train, self.result_test = df_processor.get_data()

        # Apply feature selection
        self._apply_feature_selection()

    def _apply_feature_selection(self):
        # Store all feature columns before selection
        self.feature_columns = self.input_data_train.columns.tolist()

        # Fit selector on training data and transform both train and test
        selected_train = self.selector.fit_transform(self.input_data_train, self.result_train)
        selected_test = self.selector.transform(self.input_data_test)

        # Get the indices and names of the selected features
        selected_indices = self.selector.get_support(indices=True)
        self.selected_columns = self.input_data_train.columns[selected_indices].tolist()

        # Convert back to DataFrame with selected columns
        self.input_data_train_selected = pd.DataFrame(
            selected_train,
            columns=self.selected_columns,
            index=self.input_data_train.index
        )
        self.input_data_test_selected = pd.DataFrame(
            selected_test,
            columns=self.selected_columns,
            index=self.input_data_test.index
        )

    def train(self):
        # Normalize the selected features
        X_train_normalized = self.normalizer.fit_transform(self.input_data_train_selected)
        X_test_normalized = self.normalizer.transform(self.input_data_test_selected)

        # Perform cross-validation
        cv_scores = cross_val_score(self.model, X_train_normalized, self.result_train, cv=5, scoring='accuracy')
        print("Cross-Validation Scores:", cv_scores)
        print("Mean CV Score:", cv_scores.mean())
        print("Standard Deviation of CV Scores:", cv_scores.std())

        # Train the model on the full training set
        self.model.fit(X_train_normalized, self.result_train)

        # Evaluate on the test set
        y_pred = self.model.predict(X_test_normalized)
        accuracy = accuracy_score(self.result_test, y_pred)
        print(f"Test Accuracy: {accuracy:.4f}")
        print("\nClassification Report:")
        print(classification_report(self.result_test, y_pred))

    def predict(self, input_data):
        # Ensure input_data is a DataFrame with the same feature columns as the original dataset
        if not isinstance(input_data, pd.DataFrame):
            input_data = pd.DataFrame([input_data], columns=self.feature_columns)

        # Apply feature selection to the new input
        input_selected = self.selector.transform(input_data)
        input_selected_df = pd.DataFrame(input_selected, columns=self.selected_columns)

        # Normalize the input data
        input_normalized = self.normalizer.transform(input_selected_df)

        # Make prediction
        prediction = self.model.predict(input_normalized)
        probability = self.model.predict_proba(input_normalized)

        return {
            'prediction': int(prediction[0]),
            'probability': probability[0].tolist()  # [prob_no_alzheimers, prob_alzheimers]
        }