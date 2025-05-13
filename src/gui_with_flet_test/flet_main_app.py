# --- START OF FILE src/flet_main_app.py ---
import flet as ft
import os
import sys
import traceback
import time
import matplotlib
matplotlib.use('Agg') # Non-interactive backend for generating images
import matplotlib.pyplot as plt
import numpy as np
import io
import base64

# --- Add src directory and project root to Python path ---
current_script_path = os.path.dirname(os.path.abspath(__file__))
# Assuming flet_main_app.py is directly in 'src/'
src_dir = current_script_path
project_root = os.path.dirname(src_dir)

if src_dir not in sys.path:
    sys.path.append(src_dir)
if project_root not in sys.path: # For model data path if relative to project root
    sys.path.append(project_root)

# --- Model Imports ---
try:
    from model.TrainedModel import TrainedModel
    from model.Model import AlzheimerModel
    from model.ModelAccuracy import ModelAccuracy
    from model.Dataframe import DataFrame # For dataset path if needed by model
except ImportError as e:
    print(f"CRITICAL ERROR: Could not import model classes: {e}")
    print(f"Ensure 'model' directory is in: {src_dir} or {project_root}")
    print(f"Current sys.path: {sys.path}")
    sys.exit(1)

# --- Constants for Styling (Black & White Theme) ---
APP_BACKGROUND_COLOR = "#1E1E1E"
PRIMARY_TEXT_COLOR = "#E0E0E0"
SECONDARY_TEXT_COLOR = "#A0A0A0"
BUTTON_BACKGROUND_COLOR = "#333333"
BUTTON_HOVER_COLOR = "#4A4A4A"
BUTTON_BORDER_COLOR = "#AAAAAA"
INPUT_BACKGROUND_COLOR = "#282828"
INPUT_BORDER_COLOR = "#777777"
INPUT_FOCUSED_BORDER_COLOR = PRIMARY_TEXT_COLOR
ACCENT_COLOR_POSITIVE = "#2ECC71"
ACCENT_COLOR_NEGATIVE = "#E53935"
ACCENT_COLOR_NEUTRAL = "#6CA0DC"
WHITE_COLOR = "#FFFFFF"
BLACK_COLOR = "#000000"
DIALOG_BACKGROUND_COLOR = "#2C2C2C"

# --- Global App State ---
class AppState:
    def __init__(self):
        self.model_manager = TrainedModel()
        self.available_models = []
        self.selected_model_name: str | None = None
        self.loaded_alzheimer_model: AlzheimerModel | None = None
        self.metrics_cache = {} # Cache computed metrics: {model_name: metrics_dict}

    def refresh_available_models(self):
        self.available_models = self.model_manager.list_models()
        return self.available_models

    def select_and_load_model(self, page: ft.Page, model_name: str | None) -> bool:
        if not model_name or model_name == "No model selected":
            self.selected_model_name = None
            self.loaded_alzheimer_model = None
            if page: page.show_snack_bar(ft.SnackBar(ft.Text("No model selected."), open=True))
            return False
        
        if self.selected_model_name == model_name and self.loaded_alzheimer_model:
            if page: page.show_snack_bar(ft.SnackBar(ft.Text(f"Model '{model_name}' is already active."), open=True))
            return True

        try:
            if page: page.show_dialog(LoadingDialog("Loading model..."))
            self.loaded_alzheimer_model = AlzheimerModel(model_name=model_name) # Blocking call
            if self.loaded_alzheimer_model.model is None:
                raise ValueError("Core ML model component failed to load properly.")
            self.selected_model_name = model_name
            self.metrics_cache.pop(model_name, None) # Clear old metrics if any
            if page: 
                page.close_dialog()
                page.show_snack_bar(ft.SnackBar(ft.Text(f"Model '{model_name}' loaded successfully!"), open=True, bgcolor=ACCENT_COLOR_POSITIVE))
            return True
        except Exception as e:
            self.selected_model_name = None
            self.loaded_alzheimer_model = None
            if page:
                page.close_dialog()
                page.show_snack_bar(ft.SnackBar(ft.Text(f"Error loading model '{model_name}': {str(e)[:100]}...", max_lines=2), open=True, bgcolor=ACCENT_COLOR_NEGATIVE))
            traceback.print_exc()
            return False

    def get_model_metrics(self, page: ft.Page, model_name: str) -> dict | None:
        if model_name in self.metrics_cache:
            return self.metrics_cache[model_name]
        if not self.loaded_alzheimer_model or self.selected_model_name != model_name:
            # Attempt to load if not current (e.g. deep link)
            if not self.select_and_load_model(page, model_name):
                return None
        
        try:
            if page: page.show_dialog(LoadingDialog("Computing metrics..."))
            accuracy_calculator = ModelAccuracy(model_name) # Relies on AlzheimerModel to be loaded
            metrics = accuracy_calculator.compute_metrics() # Blocking call
            self.metrics_cache[model_name] = metrics
            if page: page.close_dialog()
            return metrics
        except Exception as e:
            if page:
                page.close_dialog()
                page.show_snack_bar(ft.SnackBar(ft.Text(f"Error getting metrics for '{model_name}': {str(e)[:100]}...", max_lines=2), open=True, bgcolor=ACCENT_COLOR_NEGATIVE))
            traceback.print_exc()
            return None

app_state = AppState() # Global instance

# --- Reusable UI Components / Helpers ---

class LoadingDialog(ft.AlertDialog):
    def __init__(self, text="Loading..."):
        super().__init__(
            modal=True,
            title=ft.Text("Please Wait", color=PRIMARY_TEXT_COLOR),
            content=ft.Row(
                [ft.ProgressRing(color=ACCENT_COLOR_NEUTRAL), ft.Text(text, color=SECONDARY_TEXT_COLOR)],
                spacing=15, alignment=ft.MainAxisAlignment.START
            ),
            bgcolor=DIALOG_BACKGROUND_COLOR,
            shape=ft.RoundedRectangleBorder(radius=10)
        )

def create_styled_button(text: str, on_click, width=400, primary=False, destructive=False):
    bgcolor = BUTTON_BACKGROUND_COLOR
    hover_color = BUTTON_HOVER_COLOR
    text_color = PRIMARY_TEXT_COLOR
    if primary:
        bgcolor = ACCENT_COLOR_NEUTRAL
        hover_color = ft.colors.with_opacity(0.8, ACCENT_COLOR_NEUTRAL) if hasattr(ft.colors, "with_opacity") else ACCENT_COLOR_NEUTRAL # Guard
        text_color = WHITE_COLOR
    elif destructive:
        bgcolor = ACCENT_COLOR_NEGATIVE
        hover_color = ft.colors.with_opacity(0.8, ACCENT_COLOR_NEGATIVE) if hasattr(ft.colors, "with_opacity") else ACCENT_COLOR_NEGATIVE
        text_color = WHITE_COLOR

    return ft.ElevatedButton(
        content=ft.Text(text, size=16, weight=ft.FontWeight.W_600, color=text_color),
        on_click=on_click,
        style=ft.ButtonStyle(
            shape=ft.RoundedRectangleBorder(radius=8),
            padding=ft.padding.symmetric(vertical=18, horizontal=30),
            bgcolor={"": bgcolor, ft.MaterialState.HOVERED: hover_color},
            side={"": ft.BorderSide(1, BUTTON_BORDER_COLOR)}
        ),
        width=width,
        animate_scale=ft.Animation(200, ft.AnimationCurve.EASE_OUT),
        scale=ft.transform.Scale(scale=1),
        on_hover=lambda e: animate_control_scale(e.control, 1.03 if e.data == "true" else 1.0),
    )

def animate_control_scale(control: ft.Control, scale_value: float):
    if hasattr(control, 'scale'):
        control.scale = ft.transform.Scale(scale=scale_value)
        if control.page: control.page.update() # Request update if page is available

# --- Plotting Helper (Matplotlib to Base64 Image) ---
def create_plot_image_base64(plot_type: str, metrics: dict, model_name: str) -> str | None:
    # ... (Implementation from previous response, ensure using defined color constants) ...
    # (This function is quite long, so I'll assume it's correctly implemented as before)
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(7, 5.5), dpi=100) # Slightly larger for clarity
    plt.rcParams.update({
        "text.color": PRIMARY_TEXT_COLOR, "axes.labelcolor": SECONDARY_TEXT_COLOR,
        "xtick.color": SECONDARY_TEXT_COLOR, "ytick.color": SECONDARY_TEXT_COLOR,
        "axes.edgecolor": BUTTON_BORDER_COLOR, "figure.facecolor": APP_BACKGROUND_COLOR,
        "axes.facecolor": INPUT_BACKGROUND_COLOR, "legend.facecolor": INPUT_BACKGROUND_COLOR,
        "legend.edgecolor": BUTTON_BORDER_COLOR, "legend.labelcolor": PRIMARY_TEXT_COLOR
    })
    try:
        if plot_type == "test_accuracy_bar":
            acc = metrics.get('test_accuracy', 0.0)
            mean_cv = metrics.get('cv_scores', {}).get('mean', 0.0)
            data_to_plot = {'Test Accuracy': acc, 'Mean CV Score': mean_cv}
            names = list(data_to_plot.keys())
            values = list(data_to_plot.values())
            bars = ax.bar(names, values, color=[ACCENT_COLOR_NEUTRAL, ACCENT_COLOR_POSITIVE], width=0.5)
            ax.set_ylim(0, 1.05)
            ax.set_ylabel("Accuracy Score")
            ax.set_title("Accuracy Overview", color=PRIMARY_TEXT_COLOR)
            for bar in bars: # Add value labels on bars
                yval = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2.0, yval + 0.01, f'{yval:.3f}', ha='center', va='bottom', color=PRIMARY_TEXT_COLOR)

        elif plot_type == "confusion_matrix_heatmap":
            cm = np.array(metrics.get('confusion_matrix', [[0,0],[0,0]]))
            cax = ax.matshow(cm, cmap='Greys') # Matplotlib 'Greys' cmap
            # fig.colorbar(cax) # Colorbar can be busy for simple 2x2
            ax.set_xlabel("Predicted Label", color=SECONDARY_TEXT_COLOR)
            ax.set_ylabel("True Label", color=SECONDARY_TEXT_COLOR)
            ax.set_xticks(np.arange(cm.shape[1])); ax.set_yticks(np.arange(cm.shape[0]))
            ax.set_xticklabels(["No Alz.", "Alzheimer's"], color=PRIMARY_TEXT_COLOR)
            ax.set_yticklabels(["No Alz.", "Alzheimer's"], color=PRIMARY_TEXT_COLOR)
            ax.xaxis.set_ticks_position('bottom')
            ax.set_title("Confusion Matrix", color=PRIMARY_TEXT_COLOR)
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    text_color = BLACK_COLOR if cm[i,j] > np.max(cm) * 0.6 else PRIMARY_TEXT_COLOR
                    ax.text(j, i, str(cm[i,j]), va='center', ha='center', color=text_color, fontsize=12, weight='bold')
        
        elif plot_type == "cv_scores_bar":
            cv_scores = metrics.get('cv_scores', {}).get('scores', [0.0]*5)
            folds = [f"Fold {i+1}" for i in range(len(cv_scores))]
            bars = ax.bar(folds, cv_scores, color=ACCENT_COLOR_NEUTRAL, width=0.6)
            ax.set_ylim(0,1.05); ax.set_ylabel("Accuracy Score")
            ax.set_title("Cross-Validation Scores (Training)", color=PRIMARY_TEXT_COLOR)
            plt.xticks(rotation=30, ha="right", color=PRIMARY_TEXT_COLOR)
            for bar in bars:
                yval = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2.0, yval + 0.01, f'{yval:.3f}', ha='center', va='bottom', color=PRIMARY_TEXT_COLOR)

        elif plot_type == "learning_curve":
            lc_data = metrics.get('learning_curve')
            if lc_data and lc_data.get('train_sizes_abs'):
                ts = np.array(lc_data['train_sizes_abs'])
                tm = np.array(lc_data.get('train_scores_mean',[])); ts_std = np.array(lc_data.get('train_scores_std',[]))
                vm = np.array(lc_data.get('valid_scores_mean',[])); vs_std = np.array(lc_data.get('valid_scores_std',[]))
                if all(a.size > 0 for a in [ts,tm,ts_std,vm,vs_std]):
                    ax.plot(ts, tm, 'o-', color=ACCENT_COLOR_NEUTRAL, label="Training score")
                    ax.fill_between(ts, tm - ts_std, tm + ts_std, alpha=0.15, color=ACCENT_COLOR_NEUTRAL)
                    ax.plot(ts, vm, 'o-', color=ACCENT_COLOR_POSITIVE, label="Validation score")
                    ax.fill_between(ts, vm - vs_std, vm + vs_std, alpha=0.15, color=ACCENT_COLOR_POSITIVE)
                    ax.set_xlabel("Training Samples"); ax.set_ylabel("Accuracy Score")
                    ax.legend(loc="best"); ax.set_title("Learning Curve", color=PRIMARY_TEXT_COLOR)
                    ax.grid(True, linestyle=':', alpha=0.4, color=SECONDARY_TEXT_COLOR)
                    ax.set_ylim(min(np.min(tm-ts_std), np.min(vm-vs_std)) - 0.05 if len(tm)>0 and len(vm)>0 else 0, 1.05)

                else: ax.text(0.5,0.5,"LC data incomplete.",ha='center',va='center',color=SECONDARY_TEXT_COLOR)
            else: ax.text(0.5,0.5,"LC data not available.",ha='center',va='center',color=SECONDARY_TEXT_COLOR)
        else: ax.text(0.5,0.5,f"Plot '{plot_type}' undefined.",ha='center',va='center',color=SECONDARY_TEXT_COLOR)
        
        plt.tight_layout(pad=1.5)
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format="png", facecolor=fig.get_facecolor(), bbox_inches='tight')
        img_buffer.seek(0); plt.close(fig)
        return base64.b64encode(img_buffer.getvalue()).decode("utf-8")
    except Exception as e:
        print(f"Error generating plot '{plot_type}': {e}"); traceback.print_exc()
        if 'fig' in locals() and plt.fignum_exists(fig.number): plt.close(fig)
        return None

# --- Flet View Classes (inheriting ft.Column) ---

class MainScreenView(ft.Column):
    def __init__(self, page: ft.Page, navigate_func):
        super().__init__(
            # Basic properties for a Column that should fill space
            alignment=ft.MainAxisAlignment.CENTER,
            horizontal_alignment=ft.CrossAxisAlignment.CENTER,
            spacing=20, # Add some spacing
            expand=True # Crucial for it to take up space in the View
        )
        self.page = page
        self.navigate = navigate_func
        
        # --- EXTREMELY SIMPLIFIED UI FOR DEBUGGING ---
        self.controls = [
            ft.Text("Main Screen View - Test", size=30, color=PRIMARY_TEXT_COLOR),
            ft.ElevatedButton("Go to Train (Test)", on_click=lambda _: self.navigate("/train"))
        ]
        print("DEBUG: MainScreenView initialized with SIMPLIFIED controls.") # Debug print

    def _on_model_select(self, e):
        app_state.select_and_load_model(self.page, self.model_dd.value)
        self.delete_btn.visible = bool(app_state.selected_model_name and app_state.selected_model_name != "No model selected")
        self.update() # Update this view

    def _confirm_delete(self, e):
        # ... (Confirmation Dialog logic as in previous MainView, adapted for Flet) ...
        if not app_state.selected_model_name or app_state.selected_model_name == "No model selected": return
        confirm_dialog = ft.AlertDialog(
            modal=True, title=ft.Text("Confirm Deletion"),
            content=ft.Text(f"Delete model '{app_state.selected_model_name}'? This cannot be undone."),
            actions=[
                ft.TextButton("DELETE", on_click=self._do_delete, style=ft.ButtonStyle(color=ACCENT_COLOR_NEGATIVE)),
                ft.TextButton("Cancel", on_click=lambda _: self._close_dialog(confirm_dialog))
            ], bgcolor=DIALOG_BACKGROUND_COLOR
        )
        self.page.dialog = confirm_dialog; confirm_dialog.open = True; self.page.update()

    def _do_delete(self, e):
        # ... (Deletion logic as in previous MainView) ...
        name = app_state.selected_model_name
        try:
            if app_state.model_manager.delete_model(name):
                self.page.show_snack_bar(ft.SnackBar(ft.Text(f"Model '{name}' deleted."), bgcolor=ACCENT_COLOR_POSITIVE, open=True))
                app_state.selected_model_name = None; app_state.loaded_alzheimer_model = None
                app_state.refresh_available_models()
                self.model_dd.options = [ft.dropdown.Option("No model selected")] + [ft.dropdown.Option(n) for n in app_state.available_models]
                self.model_dd.value = "No model selected"; self.delete_btn.visible = False
            else: self.page.show_snack_bar(ft.SnackBar(ft.Text(f"Failed to delete '{name}'."), bgcolor=ACCENT_COLOR_NEGATIVE, open=True))
        except Exception as ex: self.page.show_snack_bar(ft.SnackBar(ft.Text(f"Error deleting: {ex}"), bgcolor=ACCENT_COLOR_NEGATIVE, open=True))
        self._close_dialog(self.page.dialog); self.update()

    def _close_dialog(self, dialog: ft.AlertDialog):
        if dialog: dialog.open = False; self.page.dialog = None; self.page.update()
    
    def _nav_to_predict(self):
        if not app_state.loaded_alzheimer_model:
            self.page.show_snack_bar(ft.SnackBar(ft.Text("Select a model to make predictions."), bgcolor=ACCENT_COLOR_NEGATIVE, open=True))
            return
        self.navigate("/prediction")

    def _nav_to_details(self):
        if not app_state.selected_model_name or app_state.selected_model_name == "No model selected":
            self.page.show_snack_bar(ft.SnackBar(ft.Text("Select a model to view details."), bgcolor=ACCENT_COLOR_NEGATIVE, open=True))
            return
        self.navigate(f"/details/{app_state.selected_model_name}")


class TrainModelView(ft.Column):
    # ... (Implementation similar to FletTrainModelView, using create_styled_button) ...
    # (This is getting very long, so I'll sketch it)
    def __init__(self, page: ft.Page, navigate_func):
        super().__init__(horizontal_alignment=ft.CrossAxisAlignment.CENTER, spacing=20, expand=True, scroll=ft.ScrollMode.ADAPTIVE)
        self.page = page; self.navigate = navigate_func
        self.name_field = ft.TextField(label="New Model Name", width=400, border_color=INPUT_BORDER_COLOR, focused_border_color=INPUT_FOCUSED_BORDER_COLOR, bgcolor=INPUT_BACKGROUND_COLOR, color=PRIMARY_TEXT_COLOR)
        self.status_text = ft.Text("", color=SECONDARY_TEXT_COLOR, text_align=ft.TextAlign.CENTER)
        self.train_btn = create_styled_button("Start Training", self._start_train, primary=True)
        self.controls = [
            ft.Row([ft.IconButton(ft.icons.ARROW_BACK_IOS_NEW, on_click=lambda _: self.navigate("/"), tooltip="Back"),
                    ft.Text("Train New Model", size=28, weight=ft.FontWeight.BOLD, expand=True, text_align=ft.TextAlign.CENTER)],
                   alignment=ft.MainAxisAlignment.SPACE_BETWEEN, vertical_alignment=ft.CrossAxisAlignment.CENTER),
            ft.Divider(color=BUTTON_BORDER_COLOR),
            ft.Container(height=20), self.name_field, self.train_btn, 
            ft.Container(self.status_text, padding=10, border_radius=5, visible=False, border=ft.border.all(1, SECONDARY_TEXT_COLOR))
        ]
    def _start_train(self, e):
        name = self.name_field.value.strip()
        status_container = self.controls[-1] # Assuming status_text's container is last
        if not name: 
            self.status_text.value="Name required."; self.status_text.color=ACCENT_COLOR_NEGATIVE
            status_container.border.color = ACCENT_COLOR_NEGATIVE; status_container.visible=True; self.update(); return
        # Add other name validations (chars, exists)
        if name in app_state.model_manager.list_models():
            self.status_text.value=f"'{name}' already exists."; self.status_text.color=ACCENT_COLOR_NEGATIVE
            status_container.border.color = ACCENT_COLOR_NEGATIVE; status_container.visible=True; self.update(); return

        self.train_btn.disabled = True; self.name_field.disabled = True
        self.status_text.value = f"Training '{name}'...\nThis may take a moment."; self.status_text.color = ACCENT_COLOR_NEUTRAL
        status_container.border.color = ACCENT_COLOR_NEUTRAL; status_container.visible=True; self.update(); self.page.update()
        try:
            # IMPORTANT: Flet is single-threaded by default. Long operations block UI.
            # For real app, run training in a separate thread and update UI via page.run_thread_safe()
            # For this example, it will block.
            time.sleep(0.1) # Give UI a tiny chance to update before blocking
            new_model = AlzheimerModel(); results = new_model.train(name)
            app_state.refresh_available_models(); app_state.select_and_load_model(self.page, name)
            self.status_text.value = f"'{name}' trained!\nAcc: {results.get('test_accuracy',0):.3f}, CV: {results.get('cv_data',{}).get('mean_cv_score',0):.3f}"
            self.status_text.color = ACCENT_COLOR_POSITIVE; status_container.border.color = ACCENT_COLOR_POSITIVE
            self.page.show_snack_bar(ft.SnackBar(ft.Text("Training complete!"), bgcolor=ACCENT_COLOR_POSITIVE, open=True))
            # Add a button to return or auto-return
            self.controls.append(create_styled_button("Back to Main", lambda _: self.navigate("/")))
        except Exception as ex:
            self.status_text.value = f"Training error: {str(ex)[:150]}"; self.status_text.color = ACCENT_COLOR_NEGATIVE
            status_container.border.color = ACCENT_COLOR_NEGATIVE; traceback.print_exc()
        finally:
            self.train_btn.disabled = False; self.name_field.disabled = False; self.update()


class ModelDetailsView(ft.Column):
    # ... (Implementation similar to FletAccuracyDetailsView, using create_plot_image_base64) ...
    # (This is also very long, so sketching structure)
    def __init__(self, page: ft.Page, navigate_func, model_name: str):
        super().__init__(expand=True, scroll=ft.ScrollMode.ADAPTIVE, spacing=0) # No spacing for Tabs to fill
        self.page = page; self.navigate = navigate_func; self.model_name = model_name
        self.metrics = None
        self.tabs_control = ft.Tabs(tabs=[], expand=True, animation_duration=250,
                                    label_color=ACCENT_COLOR_NEUTRAL, unselected_label_color=SECONDARY_TEXT_COLOR,
                                    indicator_color=ACCENT_COLOR_NEUTRAL, divider_color=BUTTON_BORDER_COLOR)
        self.controls = [
            ft.Row([ft.IconButton(ft.icons.ARROW_BACK_IOS_NEW, on_click=lambda _: self.navigate("/"), tooltip="Back"),
                    ft.Text(f"Details: {model_name}", size=28, weight=ft.FontWeight.BOLD, expand=True, text_align=ft.TextAlign.CENTER)],
                   alignment=ft.MainAxisAlignment.SPACE_BETWEEN, vertical_alignment=ft.CrossAxisAlignment.CENTER),
            ft.Divider(color=BUTTON_BORDER_COLOR),
            ft.Container(self.tabs_control, expand=True, padding=ft.padding.only(top=5)) # Container for Tabs
        ]
        self._load_metrics_and_build()

    def _load_metrics_and_build(self):
        self.metrics = app_state.get_model_metrics(self.page, self.model_name)
        if not self.metrics:
            self.tabs_control.tabs = [ft.Tab(text="Error", content=ft.Text("Could not load metrics.", color=ACCENT_COLOR_NEGATIVE))]
            self.update(); return
        
        tabs_list = []
        plot_configs = [
            ("Overview", "test_accuracy_bar", None), # None means no specific text, plot is main content
            ("Conf. Matrix", "confusion_matrix_heatmap", f"TN:{self.metrics['confusion_matrix'][0][0]} FP:{self.metrics['confusion_matrix'][0][1]}\nFN:{self.metrics['confusion_matrix'][1][0]} TP:{self.metrics['confusion_matrix'][1][1]}"),
            ("CV Scores", "cv_scores_bar", f"Mean: {self.metrics['cv_scores']['mean']:.3f}, Std: {self.metrics['cv_scores']['std']:.3f}"),
            ("Learning Curve", "learning_curve", None),
        ]
        for tab_name, plot_key, text_info in plot_configs:
            img_b64 = create_plot_image_base64(plot_key, self.metrics, self.model_name)
            plot_content = [ft.Image(src_base64=img_b64, fit=ft.ImageFit.CONTAIN, expand=True) if img_b64 else ft.Text("Plot unavailable.", color=SECONDARY_TEXT_COLOR)]
            if text_info: plot_content.append(ft.Text(text_info, color=SECONDARY_TEXT_COLOR, text_align=ft.TextAlign.CENTER, font_family="monospace"))
            tabs_list.append(ft.Tab(text=tab_name, content=ft.Column(plot_content, scroll=ft.ScrollMode.ADAPTIVE, horizontal_alignment=ft.CrossAxisAlignment.CENTER, spacing=10, expand=True)))

        # Classification Report (Text)
        # ... (Generate report_str as in previous FletAccuracyDetailsView) ...
        report_str = "Classification Report:\n..." # Placeholder
        tabs_list.append(ft.Tab(text="Report", content=ft.Container(ft.Text(report_str, selectable=True, font_family="monospace"), padding=15, expand=True, scroll=ft.ScrollMode.ALWAYS)))
        # Feature Importance (Text)
        # ... (Generate fi_str as in previous FletAccuracyDetailsView) ...
        fi_str = "Feature Importances:\n..." # Placeholder
        tabs_list.append(ft.Tab(text="Importance", content=ft.Container(ft.Text(fi_str, selectable=True, font_family="monospace"), padding=15, expand=True, scroll=ft.ScrollMode.ALWAYS)))

        self.tabs_control.tabs = tabs_list
        self.update()


class PredictionSessionView(ft.Column):
    # ... (__init__ method) ...

    def _prepare_questions_data(self): # Renamed from _prepare_questions
        if not self.model: # Check if model instance exists
            self.page.show_snack_bar(ft.SnackBar(ft.Text("Model not loaded for prediction setup."), bgcolor=ACCENT_COLOR_NEGATIVE, open=True))
            return

        all_features = self.model.feature_columns if self.model.feature_columns else []
        selected_by_model = self.model.selected_columns if self.model.selected_columns else []
        
        # vvvvv PASTE YOUR COPIED DICTIONARY HERE vvvvv
        feature_to_question_map = {
            "Age": {"q": "Patient's age?", "type": "text", "range": (20, 100), "unit": "years"}, # Added unit for consistency
            "Gender": {"q": "Patient's gender?", "type": "binary_radio", "options": {"Male": 0, "Female": 1}},
            "Ethnicity": {"q": "Patient's ethnicity?", "type": "categorical_radio",
                          "options": {"Caucasian": 0, "African American": 1, "Asian": 2, "Other": 3}},
            "EducationLevel": {"q": "Highest education level?", "type": "categorical_radio",
                               "options": {"None": 0, "High School": 1, "Bachelor's": 2, "Higher": 3}},
            "BMI": {"q": "Body Mass Index (BMI)?", "type": "text", "range": (15.0, 50.0), "hint": "e.g., 22.5"}, # Added hint
            "Smoking": {"q": "Is the patient a smoker?", "type": "binary_radio", "options": {"No": 0, "Yes": 1}},
            "AlcoholConsumption": {"q": "Weekly alcohol consumption (units)?", "type": "text", "range": (0, 50), "unit":"units"},
            "PhysicalActivity": {"q": "Weekly physical activity (hours)?", "type": "text", "range": (0.0, 25.0), "unit":"hours"},
            "DietQuality": {"q": "Rate diet quality (0-10, higher is better)?", "type": "text", "range": (0, 10)},
            "SleepQuality": {"q": "Rate sleep quality (4-10, higher is better)?", "type": "text", "range": (4, 10)},
            "FamilyHistoryAlzheimers": {"q": "Family history of Alzheimer's?", "type": "binary_radio", "options": {"No": 0, "Yes": 1}},
            "CardiovascularDisease": {"q": "History of cardiovascular disease?", "type": "binary_radio", "options": {"No": 0, "Yes": 1}},
            "Diabetes": {"q": "Diagnosed with diabetes?", "type": "binary_radio", "options": {"No": 0, "Yes": 1}},
            "Depression": {"q": "Diagnosed with depression?", "type": "binary_radio", "options": {"No": 0, "Yes": 1}},
            "HeadInjury": {"q": "History of significant head injury?", "type": "binary_radio", "options": {"No": 0, "Yes": 1}},
            "Hypertension": {"q": "Diagnosed with hypertension?", "type": "binary_radio", "options": {"No": 0, "Yes": 1}},
            "SystolicBP": {"q": "Systolic blood pressure (mmHg)?", "type": "text", "range": (70, 250), "unit":"mmHg"},
            "DiastolicBP": {"q": "Diastolic blood pressure (mmHg)?", "type": "text", "range": (40, 150), "unit":"mmHg"},
            "CholesterolTotal": {"q": "Total cholesterol (mg/dL)?", "type": "text", "range": (100, 400), "unit":"mg/dL"},
            "CholesterolLDL": {"q": "LDL (bad) cholesterol (mg/dL)?", "type": "text", "range": (30, 300), "unit":"mg/dL"},
            "CholesterolHDL": {"q": "HDL (good) cholesterol (mg/dL)?", "type": "text", "range": (10, 150), "unit":"mg/dL"},
            "CholesterolTriglycerides": {"q": "Triglycerides level (mg/dL)?", "type": "text", "range": (30, 600), "unit":"mg/dL"},
            "MMSE": {"q": "Mini-Mental State Exam (MMSE) score?", "type": "text", "range": (0, 30), "hint": "0-30, lower = impairment"},
            "FunctionalAssessment": {"q": "Functional assessment score?", "type": "text", "range": (0, 10), "hint": "0-10, lower = impairment"},
            "MemoryComplaints": {"q": "Patient has memory complaints?", "type": "binary_radio", "options": {"No": 0, "Yes": 1}},
            "BehavioralProblems": {"q": "Patient exhibits behavioral problems?", "type": "binary_radio", "options": {"No": 0, "Yes": 1}},
            "ADL": {"q": "Activities of Daily Living (ADL) score?", "type": "text", "range": (0, 10), "hint": "0-10, lower = impairment"},
            "Confusion": {"q": "Experiences episodes of confusion?", "type": "binary_radio", "options": {"No": 0, "Yes": 1}},
            "Disorientation": {"q": "Experiences disorientation?", "type": "binary_radio", "options": {"No": 0, "Yes": 1}},
            "PersonalityChanges": {"q": "Noticeable personality changes?", "type": "binary_radio", "options": {"No": 0, "Yes": 1}},
            "DifficultyCompletingTasks": {"q": "Difficulty completing familiar tasks?", "type": "binary_radio", "options": {"No": 0, "Yes": 1}},
            "Forgetfulness": {"q": "Significant forgetfulness?", "type": "binary_radio", "options": {"No": 0, "Yes": 1}},
        }
        # ^^^^^ PASTE YOUR COPIED DICTIONARY HERE ^^^^^

        self.questions = [] # Changed from self.questions_data
        for feature_name in all_features:
            q_info = feature_to_question_map.get(feature_name)
            if q_info:
                is_mandatory = feature_name in selected_by_model
                self.questions.append({
                    "feature": feature_name,
                    "text": q_info["q"], # The question text
                    "type": q_info["type"], # 'text', 'binary_radio', 'categorical_radio'
                    "options": q_info.get("options"), # For radio types
                    "range": q_info.get("range"),   # For text (numeric) types
                    "unit": q_info.get("unit"),     # Optional unit display
                    "hint": q_info.get("hint"),     # Optional hint text
                    "mandatory": is_mandatory,
                    "control": None # Placeholder for the Flet input control
                })
            # Initialize answer for all features model expects, even if not in map (model will impute)
            self.answers[feature_name] = None
        
        if not self.questions and all_features:
            print(f"Warning: Model expects features ({len(all_features)}), but no questions were generated. Check feature_to_question_map.")
            self.page.show_snack_bar(ft.SnackBar(ft.Text("Could not generate questions for this model."), bgcolor=ACCENT_COLOR_NEGATIVE, open=True))

    def _prepare_and_load_first_question(self):
        # ... (Simplified version of your _prepare_questions_data from PredictionView) ...
        # This needs the full feature_to_question_map
        if not self.model: self.question_text_area.value = "Model not loaded!"; self.update(); return
        # Example, assuming map exists
        feature_to_question_map = { "Age": {"q": "Patient's age?", "type": "text", "range": (20,100)}, 
                                     "Gender": {"q": "Gender?", "type": "radio", "options": {"Male":0, "Female":1}} } # Add ALL
        self.questions = []
        for f_name in self.model.feature_columns: # Iterate all expected features
            q_info = feature_to_question_map.get(f_name)
            if q_info:
                is_mandatory = f_name in (self.model.selected_columns or [])
                self.questions.append({"feature":f_name, "text":q_info["q"], "type":q_info["type"], 
                                       "options":q_info.get("options"), "range":q_info.get("range"), "mandatory":is_mandatory})
            self.answers[f_name] = None # Initialize all answers to None
        
        if self.questions: self._display_q()
        else: self.question_text_area.value = "No questions defined for this model."; self.update()


    def _display_q(self):
        if not (0 <= self.current_q_idx < len(self.questions)):
            self._show_prediction_results(); return

        q_def = self.questions[self.current_q_idx]
        self.question_text_area.value = q_def["text"]
        # Create input control dynamically (VERY simplified here)
        current_input = None
        if q_def["type"] == "text":
            current_input = ft.TextField(label="Your Answer", width=300, border_color=INPUT_BORDER_COLOR, text_align=ft.TextAlign.CENTER, bgcolor=INPUT_BACKGROUND_COLOR, color=PRIMARY_TEXT_COLOR)
            if q_def.get("range"): current_input.hint_text = f"Range: {q_def['range']}"
            if self.answers.get(q_def["feature"]) is not None: current_input.value = str(self.answers[q_def["feature"]])
        elif q_def["type"] == "radio" and q_def.get("options"):
            radios = [ft.Radio(value=str(v), label=k) for k,v in q_def["options"].items()]
            current_input = ft.RadioGroup(content=ft.Row(radios, alignment=ft.MainAxisAlignment.CENTER, spacing=15))
            if self.answers.get(q_def["feature"]) is not None: current_input.value = str(self.answers[q_def["feature"]])
        # Store ref to current input for getting value
        self.input_control_area.content = current_input 
        self.current_input_control = current_input

        self.prev_btn.disabled = (self.current_q_idx == 0)
        self.next_btn.text = "Get Prediction" if self.current_q_idx == len(self.questions) - 1 else "Next"
        self.status_text.value = f"Question {self.current_q_idx + 1} of {len(self.questions)}{' (Mandatory)' if q_def['mandatory'] else ''}"
        self.update()

    def _save_q_answer(self) -> bool:
        # ... (Simplified validation and saving) ...
        if not hasattr(self, 'current_input_control') or not self.current_input_control: return True # No input to save
        q_def = self.questions[self.current_q_idx]
        val = self.current_input_control.value
        # Basic validation
        if q_def["mandatory"] and (val is None or str(val).strip() == ""):
            self.page.show_snack_bar(ft.SnackBar(ft.Text(f"'{q_def['text']}' is mandatory."), bgcolor=ACCENT_COLOR_NEGATIVE, open=True))
            if isinstance(self.current_input_control, ft.TextField): self.current_input_control.error_text = "Required"
            return False
        
        # Type conversion and range for text (simplified)
        if q_def["type"] == "text" and val is not None and str(val).strip() != "":
            try:
                is_int = q_def.get("range") and isinstance(q_def["range"][0], int)
                num_val = int(val) if is_int else float(val)
                if q_def.get("range") and not (q_def["range"][0] <= num_val <= q_def["range"][1]):
                    if isinstance(self.current_input_control, ft.TextField): self.current_input_control.error_text = f"Out of range {q_def['range']}"
                    return False
                self.answers[q_def["feature"]] = num_val
            except ValueError:
                if isinstance(self.current_input_control, ft.TextField): self.current_input_control.error_text = "Invalid number"
                return False
        elif q_def["type"] == "radio":
             self.answers[q_def["feature"]] = int(val) if val is not None else None
        
        if isinstance(self.current_input_control, ft.TextField): self.current_input_control.error_text = None # Clear error
        return True

    def _next_q(self, e):
        if not self._save_q_answer(): self.update(); return
        if self.current_q_idx < len(self.questions) - 1:
            self.current_q_idx += 1; self._display_q()
        else: self._show_prediction_results()
        self.update()

    def _prev_q(self, e):
        self._save_q_answer() # Save without forcing validation
        if self.current_q_idx > 0:
            self.current_q_idx -= 1; self._display_q()
        self.update()

    def _show_prediction_results(self):
        # Final check for all mandatory fields before predicting
        for i, q_def in enumerate(self.questions):
            if q_def["mandatory"] and self.answers[q_def["feature"]] is None:
                self.page.show_snack_bar(ft.SnackBar(ft.Text(f"Mandatory question '{q_def['text']}' unanswered."), bgcolor=ACCENT_COLOR_NEGATIVE, open=True))
                self.current_q_idx = i; self._display_q(); self.update(); return
        try:
            results = self.model.predict(self.answers)
            pred = results['prediction']; prob = results['probability']
            res_text = "High Likelihood of Alzheimer's" if pred == 1 else "Low Likelihood of Alzheimer's"
            res_color = ACCENT_COLOR_NEGATIVE if pred == 1 else ACCENT_COLOR_POSITIVE
            
            # Replace question area with results
            self.question_text_area.value = res_text
            self.question_text_area.color = res_color
            self.question_text_area.size = 24
            self.input_control_area.content = ft.Column([
                ft.Text(f"Prob. No Alzheimer's: {prob[0]*100:.1f}%", size=16),
                ft.Text(f"Prob. Alzheimer's:    {prob[1]*100:.1f}%", size=16),
                ft.ProgressBar(value=prob[1], width=350, color=res_color, bgcolor=INPUT_BACKGROUND_COLOR, height=18, border_radius=5)
            ], horizontal_alignment=ft.CrossAxisAlignment.CENTER, spacing=8)
            self.status_text.value = "Prediction Complete."
            self.prev_btn.disabled = True; self.next_btn.text = "Finish"; 
            self.next_btn.on_click = lambda _: self.navigate("/") # Navigate home
        except Exception as ex:
            self.question_text_area.value = "Prediction Error!"; self.question_text_area.color = ACCENT_COLOR_NEGATIVE
            self.input_control_area.content = ft.Text(f"{str(ex)[:150]}", color=ACCENT_COLOR_NEGATIVE)
            traceback.print_exc()
        self.update()


def main(page: ft.Page):
    page.title = "Alzheimer's Prediction"; page.bgcolor = APP_BACKGROUND_COLOR
    # ... (other page settings as before) ...
    print("DEBUG: Flet main function started.")

    def navigate(route: str):
        print(f"DEBUG: Navigating to: {route}")
        page.go(route)

    def route_change_handler(e: ft.RouteChangeEvent):
        print(f"DEBUG: Route change to: {e.route}")
        page.views.clear() # Clear previous views from the stack

        current_view_content = None

        if e.route == "/":
            print("DEBUG: Creating MainScreenView instance.")
            try:
                current_view_content = MainScreenView(page, navigate) # USING SIMPLIFIED MainScreenView
                print("DEBUG: MainScreenView instance successfully created.")
            except Exception as ex_main_view:
                print(f"DEBUG: ERROR creating MainScreenView: {ex_main_view}")
                traceback.print_exc()
                current_view_content = ft.Text(f"Error in MainScreenView: {ex_main_view}", color=ACCENT_COLOR_NEGATIVE)
        
        # For now, let's make other routes just show a placeholder to ensure routing itself works
        elif e.route == "/train":
            print("DEBUG: Routing to Train (Placeholder).")
            current_view_content = ft.Column([
                ft.Text("Train Model View (Placeholder)", size=24, color=PRIMARY_TEXT_COLOR),
                create_styled_button("Back to Main", lambda _: navigate("/"))
            ], alignment=ft.MainAxisAlignment.CENTER, horizontal_alignment=ft.CrossAxisAlignment.CENTER, expand=True)

        # Add more placeholder routes if needed for testing navigation later
        # elif e.route == "/prediction": ...
        # elif e.route.startswith("/details/"): ...
            
        else: # Fallback for unknown routes
            print(f"DEBUG: Unknown route '{e.route}', showing error text.")
            current_view_content = ft.Text(f"Page not found: {e.route}", size=24, color=ACCENT_COLOR_NEGATIVE)


        if current_view_content:
            page.views.append(
                ft.View(
                    route=e.route, # The route for this view
                    controls=[current_view_content], # The main content for the view
                    bgcolor=APP_BACKGROUND_COLOR,
                    padding=20, # Padding for the entire view area
                    vertical_alignment=ft.MainAxisAlignment.START,
                    horizontal_alignment=ft.CrossAxisAlignment.STRETCH # Stretch content like Columns
                )
            )
            print(f"DEBUG: View for {e.route} appended to page.views. Total views: {len(page.views)}")
        else:
            # This case should ideally not be hit if there's a fallback for unknown routes
            print(f"DEBUG: No content generated for route {e.route}. This is unexpected.")
            page.views.append(ft.View(e.route, [ft.Text("Error: View content missing.", color=ACCENT_COLOR_NEGATIVE)]))

        try:
            page.update()
            print(f"DEBUG: page.update() called after routing to {e.route}.")
        except Exception as ex_update:
            print(f"DEBUG: ERROR during page.update(): {ex_update}")
            traceback.print_exc()


    def view_pop_handler(e: ft.ViewPopEvent):
        print(f"DEBUG: view_pop_handler called. Current views on stack: {len(page.views)}")
        # Flet's default behavior with page.go handles popping views from the stack correctly.
        # This handler is more for reacting to the pop if needed, or custom back logic.
        # If we always use page.go for back navigation, this might not be strictly necessary
        # unless we want to intercept the browser/OS back button differently.
        if len(page.views) > 1: # Ensure we are not at the root view
            page.views.pop() # Manually pop the current view
            if page.views: # If there are still views left
                top_view_route = page.views[-1].route
                print(f"DEBUG: Popping. Navigating back to: {top_view_route}")
                page.go(top_view_route) # Go to the previous view's route
            else: # Should not happen if root view is always present
                page.go("/")
        else:
            print("DEBUG: Already at the root view, cannot pop further.")
        # page.update() # page.go() will trigger an update via route_change

    page.on_route_change = route_change_handler
    page.on_view_pop = view_pop_handler
    
    print("DEBUG: Initial navigation to '/'")
    page.go("/") # Trigger the initial route

if __name__ == "__main__":
    # ... (print statements and ft.app(target=main) as before) ...
    print(f"Attempting to run Flet app. Current directory: {os.getcwd()}")
    print(f"Python sys.path includes: {src_dir}, {project_root}")
    tm_check = TrainedModel()
    print(f"TrainedModel manager will look for models in: {tm_check.model_dir}")
    if not tm_check.model_dir.exists():
        print(f"WARNING: Models directory '{tm_check.model_dir}' does not exist. Training will create it.")
    ft.app(target=main)