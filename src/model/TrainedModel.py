import joblib
from typing import List, Any, Dict
from pathlib import Path

class TrainedModel:
    def __init__(self, model_dir: str = "Models") -> None:
        script_path = Path(__file__).resolve()
        project_root = script_path.parent.parent.parent
        self.model_dir = project_root / model_dir
        self.model_dir.mkdir(parents=True, exist_ok=True)

    def _get_model_path(self, model_name: str) -> Path:
        invalid_chars = " /\\:*?\"<>|"
        replacement_char = "_"
        translation_table = str.maketrans(invalid_chars, replacement_char * len(invalid_chars))
        safe_base_name = model_name.translate(translation_table)
        safe_name = safe_base_name + ".pkl"
        return self.model_dir / safe_name

    def save_model(self, model_name: str, model: Any, normalizer: Any,
                  selector: Any, feature_columns: List[str],
                  selected_columns: List[str], learning_curve_data: Dict = None,
                  imputer: Any = None, full_data_imputer: Any = None) -> None:
        model_path = self._get_model_path(model_name)
        model_data = {
            'model': model,
            'normalizer': normalizer,
            'selector': selector,
            'feature_columns': feature_columns,
            'selected_columns': selected_columns,
            'learning_curve_data': learning_curve_data,
            'imputer': imputer,
            'full_data_imputer': full_data_imputer
        }
        try:
            joblib.dump(model_data, str(model_path))
            print(f"Model '{model_name}' saved successfully to {model_path}")
        except Exception as e:
             print(f"Error saving model '{model_name}' to {model_path}: {e}")

    def load_model(self, model_name: str):
        model_path = self._get_model_path(model_name)
        if not model_path.is_file():
            print(f"Model file not found: {model_path}")
            return None, None, None, None, None, None, None, None

        try:
            data = joblib.load(str(model_path))
            required_keys = ['model', 'normalizer', 'selector', 'feature_columns', 'selected_columns']
            if not all(key in data and data[key] is not None for key in required_keys):
                print(f"Error: Model file '{model_path}' is incomplete or corrupted (missing core components).")
                return None, None, None, None, None, None, None, None

            learning_curve_data = data.get('learning_curve_data', None)
            # Load both imputers, defaulting to None for backward compatibility
            imputer = data.get('imputer', None)
            full_data_imputer = data.get('full_data_imputer', None)

            print(f"Model '{model_name}' loaded successfully from {model_path}")
            return (
                data['model'], data['normalizer'], data['selector'],
                data['feature_columns'], data['selected_columns'],
                learning_curve_data, imputer, full_data_imputer
            )
        except Exception as e:
            print(f"Error loading model '{model_name}' from {model_path}: {e}")
            return None, None, None, None, None, None, None, None

    def list_models(self) -> List[str]:
        model_files = list(self.model_dir.glob("*.pkl"))
        model_names = [f.stem for f in model_files]
        return sorted(model_names)

    def delete_model(self, model_name: str) -> bool:
        model_path = self._get_model_path(model_name)
        try:
            if model_path.is_file():
                model_path.unlink()
                print(f"Model '{model_name}' deleted successfully from {model_path}")
                return True
            else:
                print(f"Error: Model file not found for deletion: {model_path}")
                return False
        except Exception as e:
             print(f"Error deleting model '{model_name}' from {model_path}: {e}")
             return False