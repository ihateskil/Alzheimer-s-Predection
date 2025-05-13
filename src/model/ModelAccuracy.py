from .Model import AlzheimerModel
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score
import numpy as np
import traceback

class ModelAccuracy:
    def __init__(self, model_name):
        self.model = AlzheimerModel(model_name=model_name)
        self.model_name = model_name

    def compute_metrics(self):
        self._validate_model()

        X_test_normalized = self.model.normalizer.transform(self.model.input_data_test_selected)
        y_test = self.model.result_test
        y_pred = self.model.model.predict(X_test_normalized)

        feature_importance = []
        if hasattr(self.model.selector, 'scores_') and self.model.selector.get_support().any():
             feature_importance = self.model.selector.scores_[self.model.selector.get_support()].tolist()
        else:
            feature_importance = [0.0] * len(self.model.selected_columns) if self.model.selected_columns else []
        selected_columns = self.model.selected_columns if self.model.selected_columns else []

        X_train_normalized_for_cv = self.model.normalizer.transform(self.model.input_data_train_selected)

        cv_scores_list = []
        cv_mean = 0.0
        cv_std = 0.0
        if self.model.model is not None:
            cv_scores_obj = cross_val_score(self.model.model, X_train_normalized_for_cv, self.model.result_train, cv=5, scoring='accuracy')
            cv_scores_list = cv_scores_obj.tolist()
            cv_mean = np.mean(cv_scores_list)
            cv_std = np.std(cv_scores_list)
        else:
             print("Warning: Cannot compute CV scores as model object is None in ModelAccuracy.")

        learning_curve_data = self.model.learning_curve_data

        return {
            'test_accuracy': accuracy_score(y_test, y_pred),
            'classification_report': classification_report(y_test, y_pred, output_dict=True, zero_division=0),
            'confusion_matrix': confusion_matrix(y_test, y_pred),
            'feature_importance': feature_importance,
            'selected_columns': selected_columns,
            'cv_scores': {
                'scores': cv_scores_list,
                'mean': cv_mean,
                'std': cv_std
            },
            'learning_curve': learning_curve_data
        }

    def _validate_model(self):
        if self.model.model is None or not hasattr(self.model.model, 'class_prior_'):
            raise ValueError(f"Model '{self.model_name}' is not trained or loaded properly.")

    def display_metrics(self):
        try:
            metrics = self.compute_metrics()
            print(f"\n--- Accuracy Metrics for Model: '{self.model_name}' ---")
            print(f"\nTest Set Accuracy: {metrics['test_accuracy']:.4f}")

            print("\nCross-Validation Scores (on Training Set):")
            if metrics['cv_scores']['scores']:
                for i, score in enumerate(metrics['cv_scores']['scores']):
                    print(f"  Fold {i+1}: {score:.4f}")
                print(f"  Mean CV Score: {metrics['cv_scores']['mean']:.4f}")
                print(f"  Std Dev CV Score: {metrics['cv_scores']['std']:.4f}")
            else:
                print("  Could not be computed.")

            print("\nClassification Report (on Test Set):")
            report = metrics['classification_report']
            print(f"  Overall Accuracy: {report.get('accuracy', 'N/A'):.4f}")
            print(f"  Macro Avg F1-Score: {report.get('macro avg', {}).get('f1-score', 'N/A'):.4f}")
            print(f"  Weighted Avg F1-Score: {report.get('weighted avg', {}).get('f1-score', 'N/A'):.4f}")

            for label, stats in report.items():
                if label in ['0', '1']:
                    class_name = 'No Alzheimer' if label == '0' else 'Alzheimer'
                    print(f"\n  Class {label} ({class_name}):")
                    print(f"    Precision: {stats.get('precision', 'N/A'):.4f}")
                    print(f"    Recall:    {stats.get('recall', 'N/A'):.4f}")
                    print(f"    F1-Score:  {stats.get('f1-score', 'N/A'):.4f}")
                    print(f"    Support:   {stats.get('support', 'N/A')}")

            print("\nConfusion Matrix (Test Set):")
            cm_array = metrics['confusion_matrix']
            tn, fp, fn, tp = cm_array.ravel() if cm_array.size == 4 else (0,0,0,0)
            print("         Predicted No Alz.  Predicted Alz.")
            print(f"Actual No Alz.    {tn:^10d}      {fp:^10d}")
            print(f"Actual Alz.       {fn:^10d}      {tp:^10d}")
            print(f"\n  - TN (True Negative):  {tn} (No Alzheimer correctly predicted)")
            print(f"  - FP (False Positive): {fp} (No Alzheimer predicted as Alzheimer)")
            print(f"  - FN (False Negative): {fn} (Alzheimer predicted as No Alzheimer)")
            print(f"  - TP (True Positive):  {tp} (Alzheimer correctly predicted)")

            print("\nTop Selected Features (Mutual Information Scores):")
            if metrics['selected_columns'] and metrics['feature_importance']:
                 feature_scores_sorted = sorted(zip(metrics['selected_columns'], metrics['feature_importance']), key=lambda item: item[1], reverse=True)
                 for col, imp in feature_scores_sorted:
                     print(f"  - {col}: {imp:.4f}")
            else:
                 print("  Feature importance scores not available.")

            lc_data = metrics.get('learning_curve')
            if lc_data and lc_data.get('train_sizes_abs'):
                 print("\nLearning Curve Summary:")
                 print(f"  Training Sizes: {lc_data.get('train_sizes_abs', 'N/A')}")
                 print(f"  Mean Training Scores: {[f'{s:.4f}' for s in lc_data.get('train_scores_mean', [])]}")
                 print(f"  Mean Validation Scores: {[f'{s:.4f}' for s in lc_data.get('valid_scores_mean', [])]}")
            else:
                 print("\nLearning Curve data not available or empty.")
            print("-" * 50)

        except ValueError as ve:
            print(f"Error displaying metrics for '{self.model_name}': {ve}")
        except Exception as e:
            traceback.print_exc() # Print full traceback for unexpected errors
            print(f"An unexpected error occurred while displaying metrics: {e}")
