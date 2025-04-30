import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from normalizer import Normalizer

class AlzheimerModel:
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.normalizer = Normalizer()
        self.feature_columns = None

    def train(self, df):
        # Extract features and target
        self.feature_columns = [col for col in df.columns if col != 'Diagnosis']
        X = df[self.feature_columns]
        y = df['Diagnosis']

        # Normalize the features
        X_normalized = self.normalizer.fit_transform(X)

        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X_normalized, y, test_size=0.2, random_state=42
        )

        # Perform cross-validation
        cv_scores = cross_val_score(self.model, X_train, y_train, cv=5, scoring='accuracy')
        print("Cross-Validation Scores:", cv_scores)
        print("Mean CV Score:", cv_scores.mean())
        print("Standard Deviation of CV Scores:", cv_scores.std())

        # Train the model on the full training set
        self.model.fit(X_train, y_train)

        # Evaluate on the test set
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Test Accuracy: {accuracy:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))

    def predict(self, input_data):
        # Ensure input_data is a DataFrame with the same feature columns
        if not isinstance(input_data, pd.DataFrame):
            input_data = pd.DataFrame([input_data], columns=self.feature_columns)

        # Normalize the input data
        input_normalized = self.normalizer.transform(input_data)

        # Make prediction
        prediction = self.model.predict(input_normalized)
        probability = self.model.predict_proba(input_normalized)

        return {
            'prediction': int(prediction[0]),
            'probability': probability[0].tolist()  # [prob_no_alzheimers, prob_alzheimers]
        }