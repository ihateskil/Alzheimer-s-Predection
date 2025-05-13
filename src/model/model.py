import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, learning_curve
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.impute import SimpleImputer
from .Normalizer import Normalizer
from .Dataframe import DataFrame

class AlzheimerModel:
    def __init__(self, model_name=None):
        self._initialize_components()
        self.model_name = model_name
        self._load_data()
        self.imputer = None
        self.full_data_imputer = None

        if model_name:
            self._load_model(model_name) # Attempt to load existing model
            # If loading failed and model_name became None, or if selector is still not fitted:
            if self.model_name is None or not hasattr(self.selector, "scores_"):
                print(f"DEBUG: Model '{model_name}' load issue or selector not fit, ensuring feature selection setup.")
                self._apply_feature_selection_and_imputation() # Renamed for clarity
        else:
            # This path is for when a new model is being prepared
            self._apply_feature_selection_and_imputation() # Renamed for clarity


    def _initialize_components(self):
        from .TrainedModel import TrainedModel # Local import
        self.model = None
        self.normalizer = Normalizer()
        self.selector = SelectKBest(score_func=mutual_info_classif, k=10)
        self.feature_columns = None
        self.selected_columns = None
        self.feature_scores = None
        self.learning_curve_data = None
        self.saver = TrainedModel()
        self.imputer = None # Imputer for selected features
        self.full_data_imputer = None # Imputer for all features before selection

    def _load_data(self):
        df_processor = DataFrame()
        self.input_data_train, self.input_data_test, self.result_train, self.result_test = df_processor.get_data()

    def _load_model(self, model_name):
        # Expect 8 items now, including both imputers
        loaded_model, loaded_normalizer, loaded_selector, \
        loaded_feature_columns, loaded_selected_columns, \
        loaded_lc_data, loaded_imputer, loaded_full_data_imputer = self.saver.load_model(model_name)

        if loaded_model is not None:
            self.model = loaded_model
            self.normalizer = loaded_normalizer
            self.selector = loaded_selector # fitted selector
            self.feature_columns = loaded_feature_columns
            self.selected_columns = loaded_selected_columns
            self.learning_curve_data = loaded_lc_data
            self.imputer = loaded_imputer
            self.full_data_imputer = loaded_full_data_imputer
            self.model_name = model_name
            print(f"Loaded model '{model_name}'.")

            # Validate and potentially refit imputers
            if self.full_data_imputer is None or not hasattr(self.full_data_imputer, 'statistics_'):
                print("Warning: Full data imputer not properly loaded/fitted. Refitting on training data.")
                self.full_data_imputer = SimpleImputer(strategy='mean')
                self.full_data_imputer.fit(self.input_data_train) # Fit on original training data
            
            # Impute the original training and test data using the full_data_imputer
            input_data_train_imputed = pd.DataFrame(self.full_data_imputer.transform(self.input_data_train), columns=self.feature_columns, index=self.input_data_train.index)
            input_data_test_imputed = pd.DataFrame(self.full_data_imputer.transform(self.input_data_test), columns=self.feature_columns, index=self.input_data_test.index)

            # --- Ensure selected training and test data are created using the loaded selector ---
            # Transform imputed full training data to get selected training features
            selected_train_values = self.selector.transform(input_data_train_imputed)
            self.input_data_train_selected = pd.DataFrame(
                selected_train_values,
                columns=self.selected_columns,
                index=self.input_data_train.index
            )
            # Transform imputed full test data to get selected test features
            selected_test_values = self.selector.transform(input_data_test_imputed)
            self.input_data_test_selected = pd.DataFrame(
                selected_test_values,
                columns=self.selected_columns,
                index=self.input_data_test.index
            )

            # Validate and refit selected feature imputer
            # This imputer should be consistent with the one saved, operating on already selected features
            if self.imputer is None or not hasattr(self.imputer, 'statistics_'):
                print("Warning: Selected features imputer not properly loaded/fitted. Refitting on selected training data.")
                self.imputer = SimpleImputer(strategy='mean')
                self.imputer.fit(self.input_data_train_selected)


            # Refit normalizer if necessary
            if not hasattr(self.normalizer.scaler, 'data_min_') or self.normalizer.scaler.data_min_ is None:
                print("Refitting Normalizer after loading...")
                self.normalizer.fit(self.input_data_train_selected) # Fit on selected imputed training data

            if hasattr(self.selector, 'scores_'): # Get feature scores from the loaded selector
                self.feature_scores = self.selector.scores_
        else:
            print(f"Model '{model_name}' could not be loaded or is not fitted.")
            self.model_name = None
            self._apply_feature_selection_and_imputation() # Fallback

    # Renamed for clarity and to emphasize imputation step
    def _apply_feature_selection_and_imputation(self):
        """Fits imputers, selects features, and prepares selected data subsets."""
        self.feature_columns = self.input_data_train.columns.tolist()

        # Fit and transform using full_data_imputer
        self.full_data_imputer = SimpleImputer(strategy='mean')
        self.full_data_imputer.fit(self.input_data_train)
        input_data_train_imputed = pd.DataFrame(self.full_data_imputer.transform(self.input_data_train), columns=self.feature_columns, index=self.input_data_train.index)
        input_data_test_imputed = pd.DataFrame(self.full_data_imputer.transform(self.input_data_test), columns=self.feature_columns, index=self.input_data_test.index)

        # Feature Selection on imputed data
        selected_train_values = self.selector.fit_transform(input_data_train_imputed, self.result_train)
        selected_test_values = self.selector.transform(input_data_test_imputed)
        selected_indices = self.selector.get_support(indices=True)
        self.selected_columns = self.input_data_train.columns[selected_indices].tolist()
        self.feature_scores = self.selector.scores_

        # Create DataFrames for selected data
        self.input_data_train_selected = pd.DataFrame(
            selected_train_values,
            columns=self.selected_columns,
            index=self.input_data_train.index
        )
        self.input_data_test_selected = pd.DataFrame(
            selected_test_values,
            columns=self.selected_columns,
            index=self.input_data_test.index
        )

        # Fit imputer for selected features (on already imputed-then-selected training data)
        self.imputer = SimpleImputer(strategy='mean')
        self.imputer.fit(self.input_data_train_selected)

        # Fit Normalizer on the selected (and imputed) training data
        self.normalizer.fit(self.input_data_train_selected)


    def train(self, model_name_to_train):
        # Ensure all preprocessing components are ready
        if self.selected_columns is None or self.imputer is None or self.full_data_imputer is None or self.normalizer.columns is None:
             self._apply_feature_selection_and_imputation()

        self.model = GaussianNB()
        self.model_name = model_name_to_train

        # Data for training and CV is after feature selection and imputation
        X_train_normalized = self.normalizer.transform(self.input_data_train_selected)
        X_test_normalized = self.normalizer.transform(self.input_data_test_selected)

        # Learning Curve
        train_sizes_frac = np.linspace(0.1, 1.0, 5)
        train_sizes_abs, train_scores, valid_scores = learning_curve(
            self.model, X_train_normalized, self.result_train,
            train_sizes=train_sizes_frac, cv=5, scoring='accuracy', n_jobs=-1
        )
        self.learning_curve_data = {
            'train_sizes_abs': train_sizes_abs.tolist(),
            'train_scores_mean': np.mean(train_scores, axis=1).tolist(),
            'train_scores_std': np.std(train_scores, axis=1).tolist(),
            'valid_scores_mean': np.mean(valid_scores, axis=1).tolist(),
            'valid_scores_std': np.std(valid_scores, axis=1).tolist()
        }

        # Cross-Validation
        cv_scores = cross_val_score(self.model, X_train_normalized, self.result_train, cv=5, scoring='accuracy')
        cv_data = {
            'cv_scores': cv_scores.tolist(), 'mean_cv_score': cv_scores.mean(), 'std_cv_score': cv_scores.std()
        }

        # Final Model Training
        self.model.fit(X_train_normalized, self.result_train)

        # Evaluation
        y_pred = self.model.predict(X_test_normalized)
        test_accuracy = accuracy_score(self.result_test, y_pred)
        report = classification_report(self.result_test, y_pred, output_dict=True, zero_division=0)

        # Save Model and all components
        self.saver.save_model(
            self.model_name, self.model, self.normalizer, self.selector,
            self.feature_columns, self.selected_columns, self.learning_curve_data,
            self.imputer, self.full_data_imputer # Pass both imputers
        )
        return {'cv_data': cv_data, 'test_accuracy': test_accuracy, 'classification_report': report}

    def predict(self, input_data_dict: dict):
        if self.model is None or not hasattr(self.model, 'class_prior_') or \
           self.full_data_imputer is None or self.selector is None or self.normalizer is None:
            error_msg = "Model or its preprocessors are not properly loaded/initialized for prediction."
            print(f"ERROR: {error_msg}")
            print(f"  self.model: {'Fitted' if hasattr(self.model, 'class_prior_') else self.model}")
            print(f"  self.full_data_imputer: {'Fitted' if hasattr(self.full_data_imputer, 'statistics_') else self.full_data_imputer}")
            print(f"  self.selector: {'Fitted' if hasattr(self.selector, 'scores_') else self.selector}")
            print(f"  self.normalizer: {'Fitted' if hasattr(self.normalizer.scaler, 'data_min_') else self.normalizer}")
            raise ValueError(error_msg)
        
        input_df_full = pd.DataFrame([input_data_dict]).reindex(columns=self.feature_columns)

        # Impute using full_data_imputer first
        input_df_full_imputed_values = self.full_data_imputer.transform(input_df_full)
        input_df_full_imputed = pd.DataFrame(input_df_full_imputed_values, columns=self.feature_columns, index=input_df_full.index)

        # Select features
        input_selected_values = self.selector.transform(input_df_full_imputed)
        input_selected_df = pd.DataFrame(input_selected_values, columns=self.selected_columns, index=input_df_full_imputed.index)

        # Safeguard imputation for selected features
        if input_selected_df.isnull().values.any():
            print("Warning: NaNs still detected in selected features for prediction. Applying selected feature imputer.")
            if self.imputer is None or not hasattr(self.imputer, 'statistics_'):
                print("ERROR: Selected feature imputer is not fitted! This is unexpected in predict path.")
                # Attempt to refit, but this indicates a logic flaw elsewhere or corrupted model state
                self.imputer = SimpleImputer(strategy='mean')
                # This refit uses self.input_data_train_selected which should exist and be clean
                if self.input_data_train_selected is None:
                    raise ValueError("Cannot refit selected imputer: input_data_train_selected is None.")
                self.imputer.fit(self.input_data_train_selected)
            input_selected_df_values = self.imputer.transform(input_selected_df)
            input_selected_df = pd.DataFrame(input_selected_df_values, columns=self.selected_columns, index=input_selected_df.index)

        # Normalize
        if not hasattr(self.normalizer.scaler, 'data_min_') or self.normalizer.scaler.data_min_ is None:
            print("Warning: Normalizer not fitted during prediction. Refitting on (hopefully clean) selected training data.")
            if self.input_data_train_selected is None:
                 raise ValueError("Cannot refit normalizer: input_data_train_selected is None.")
            self.normalizer.fit(self.input_data_train_selected)
        input_normalized = self.normalizer.transform(input_selected_df)

        prediction = self.model.predict(input_normalized)
        probability = self.model.predict_proba(input_normalized)

        return {
            'prediction': int(prediction[0]),
            'probability': probability[0].tolist()
        }