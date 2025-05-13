from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QMessageBox, QStackedWidget, QProgressBar
)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QFont
from model.Model import AlzheimerModel
from .QuestionWidget import QuestionWidget
import traceback

class PredictionView(QWidget):
    """
    Manages the multi-step process for user input to make a prediction.
    It displays questions sequentially using QuestionWidget and shows the final result.
    """
    prediction_finished = pyqtSignal()

    def __init__(self, main_window_instance, parent=None):
        super().__init__(parent)
        self.main_window = main_window_instance # Reference to MainWindow for navigation
        self.alzheimer_model_instance = None    # The loaded AlzheimerModel instance
        self.questions_definitions = []         # List of (feature, question_text, type, options, range, mandatory)
        self.user_answers = {}                  # Stores user's answers, keyed by feature_name
        self.current_question_idx = 0

        self._init_ui_layout()
        self._apply_view_styles()

    def _init_ui_layout(self):
        """Sets up the main layout and widgets for the PredictionView."""
        self.main_layout = QVBoxLayout(self)
        self.main_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.main_layout.setContentsMargins(40, 30, 40, 30)
        self.view_title_label = QLabel("Alzheimer's Prediction Input")
        self.view_title_label.setFont(QFont("Arial", 22, QFont.Weight.Bold))
        self.view_title_label.setStyleSheet("color: #E0E0E0; margin-bottom: 25px; margin-top: 10px;")
        self.view_title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.main_layout.addWidget(self.view_title_label)

        # StackedWidget to hold QuestionWidget or ResultsWidget
        self.question_stack = QStackedWidget()
        self.main_layout.addWidget(self.question_stack, 1)

        # Navigation buttons
        nav_button_container_layout = QHBoxLayout()
        nav_button_container_layout.setSpacing(15)

        self.exit_button = QPushButton("Return to Main Menu")
        self.exit_button.clicked.connect(self._emit_prediction_finished_signal)
        nav_button_container_layout.addWidget(self.exit_button, 0, Qt.AlignmentFlag.AlignLeft)

        nav_button_container_layout.addStretch(1)

        self.prev_question_button = QPushButton("Previous")
        self.prev_question_button.clicked.connect(self._navigate_to_previous_question)
        nav_button_container_layout.addWidget(self.prev_question_button, 0, Qt.AlignmentFlag.AlignRight)

        self.next_predict_button = QPushButton("Next")
        self.next_predict_button.clicked.connect(self._handle_next_or_predict_action)
        nav_button_container_layout.addWidget(self.next_predict_button, 0, Qt.AlignmentFlag.AlignRight)

        self.main_layout.addLayout(nav_button_container_layout)

    def _apply_view_styles(self):
        #Applies the theme styling to the elements.
        self.setStyleSheet("""
            QWidget { /* Base style for PredictionView itself */
                background-color: #1E1E1E; /* Dark grey background */
            }
            QLabel { /* Default for labels in this view, can be overridden */
                color: #E0E0E0; /* Off-white */
            }
            QPushButton {
                background-color: #333333;
                color: white;
                border: 1px solid #AAAAAA;
                border-radius: 6px;
                /* To adjust button size, change padding, min-width/height */
                padding: 10px 20px;
                min-height: 30px; /* Example min height */
                font-size: 12pt;  /* Font for navigation buttons */
            }
            QPushButton:hover {
                background-color: #4A4A4A;
                border: 1px solid white;
            }
            QPushButton:disabled {
                background-color: #252525;
                color: #666666;
                border: 1px solid #444444;
            }
            QProgressBar {
                border: 1px solid #AAAAAA;
                border-radius: 5px;
                background-color: #282828; /* Darker background for progress bar */
                text-align: center; /* Though text is usually hidden */
                color: white; /* For text if visible */
                min-height: 28px; /* Adjust progress bar height */
            }
            QProgressBar::chunk {
                background-color: #6CA0DC; /* A contrasting accent color (e.g., a calm blue) */
                /* width: 10px; */ /* Chunk width, often better to let it be dynamic */
                /* margin: 0.5px; */
                border-radius: 4px;
            }
        """)

    def reset_view_state(self):
       #Resets the view to its initial state, clearing answers and questions.
        self.user_answers = {}
        self.current_question_idx = 0
        self.questions_definitions = []
        while self.question_stack.count() > 0:
            widget = self.question_stack.widget(0)
            self.question_stack.removeWidget(widget)
            widget.deleteLater()
        self.view_title_label.setText("Alzheimer's Prediction Input")


    def _emit_prediction_finished_signal(self):
        self.prediction_finished.emit() # Signal MainWindow to switch view

    def set_model_and_start(self, alzheimer_model: AlzheimerModel):

      #  Sets the model to use for prediction and initializes the question sequence.

        self.reset_view_state() # Clear any previous state
        self.alzheimer_model_instance = alzheimer_model

        if self.alzheimer_model_instance and self.alzheimer_model_instance.model_name:
            self.view_title_label.setText(f"Predicting with: {self.alzheimer_model_instance.model_name}")
        else:
            QMessageBox.critical(self, "Model Error", "No valid Alzheimer's model instance provided.")
            self._emit_prediction_finished_signal() # Go back if no model
            return

        self._generate_question_definitions()
        if not self.questions_definitions:
            QMessageBox.critical(self, "Setup Error", "Could not prepare questions for the selected model.")
            self._emit_prediction_finished_signal()
            return

        self._display_current_question() # Show the first question

    def _generate_question_definitions(self):
       # Prepares the list of questions based on the features of the loaded model.
        
        if not self.alzheimer_model_instance: return

        all_model_features = self.alzheimer_model_instance.feature_columns or []
        selected_by_model = self.alzheimer_model_instance.selected_columns or []

        # This map defines how each feature translates to a user question
        feature_to_question_map = {
            "Age": {"q": "What is the patient's age?", "type": "text", "range": (20, 100), "unit": "years"},
            "Gender": {"q": "Patient's gender?", "type": "binary_radio", "options": {"Male": 0, "Female": 1}},
            "Ethnicity": {"q": "Patient's ethnicity?", "type": "categorical_radio",
                          "options": {"Caucasian": 0, "African American": 1, "Asian": 2, "Other": 3}},
            "EducationLevel": {"q": "Highest education level?", "type": "categorical_radio",
                               "options": {"None": 0, "High School": 1, "Bachelor's": 2, "Higher": 3}},
            "BMI": {"q": "Body Mass Index (BMI)?", "type": "text", "range": (15.0, 50.0), "hint": "e.g., 22.5"},
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

        self.questions_definitions = []
        for feature_name in all_model_features:
            if feature_name in feature_to_question_map:
                q_info = feature_to_question_map[feature_name]
                # A feature is mandatory for input if the *trained model selected it for use*.
                # Non-selected features are still asked if they are in `all_model_features`
                # because the `full_data_imputer` in `AlzheimerModel` expects all of them.
                # However, the user only *must* answer questions for selected features.
                is_mandatory_input = feature_name in selected_by_model

                self.questions_definitions.append({
                    "feature": feature_name,
                    "text": q_info["q"],
                    "type": q_info["type"],
                    "options": q_info.get("options"),
                    "range": q_info.get("range"),
                    "hint": q_info.get("hint"),
                    "unit": q_info.get("unit"),
                    "mandatory": is_mandatory_input
                })
                self.user_answers[feature_name] = None # Initialize answer
            else:
                # This means a feature expected by the model doesn't have a question defined.
                # The model's `predict` method will impute it if it's None.
                print(f"Warning: No question definition for model feature: '{feature_name}'. It will be handled by imputer if not answered.")
                self.user_answers[feature_name] = None # Needs to be in user_answers for imputation pass

    def _display_current_question(self):
       # Creates and shows the QuestionWidget for the current question index.
        while self.question_stack.count() > 0:
            widget_to_remove = self.question_stack.widget(0)
            self.question_stack.removeWidget(widget_to_remove)
            widget_to_remove.deleteLater() # to free memory

        if 0 <= self.current_question_idx < len(self.questions_definitions):
            q_def = self.questions_definitions[self.current_question_idx]
            question_widget = QuestionWidget(
                feature_name=q_def["feature"],
                question_text=q_def["text"],
                input_type=q_def["type"],
                options=q_def["options"],
                data_range=q_def["range"],
                is_mandatory=q_def["mandatory"],
                hint_text=q_def.get("hint"),
                unit_text=q_def.get("unit"),
                parent=self 
            )
            # Pre-fill answer if user is navigating back
            if self.user_answers.get(q_def["feature"]) is not None:
                question_widget.set_current_answer(self.user_answers[q_def["feature"]])

            self.question_stack.addWidget(question_widget)
            self.question_stack.setCurrentWidget(question_widget)
        self._update_navigation_buttons_state()

    def _update_navigation_buttons_state(self):
        # Updates the enabled state and text of navigation buttons.
        self.prev_question_button.setEnabled(self.current_question_idx > 0)

        # Check if the current widget is the results page or a question page
        current_widget_on_stack = self.question_stack.currentWidget()
        is_on_results_page = not isinstance(current_widget_on_stack, QuestionWidget) and current_widget_on_stack is not None

        if is_on_results_page:
            self.prev_question_button.setEnabled(False) # No going back from results
            self.next_predict_button.setText("Finish Session")
            self.next_predict_button.setEnabled(True)
        elif self.current_question_idx >= len(self.questions_definitions): # This case can happen if all questions answered and Predict was clicked, It should ideally transition to results page or error.
             self.prev_question_button.setEnabled(False)
             self.next_predict_button.setText("Finish Session")
             self.next_predict_button.setEnabled(True)
        else: # On a question page
            if self.current_question_idx == len(self.questions_definitions) - 1:
                self.next_predict_button.setText("Get Prediction")
            else:
                self.next_predict_button.setText("Next Question")
            self.next_predict_button.setEnabled(True)


    def _navigate_to_previous_question(self):
        """Saves current answer (if any) and moves to the previous question."""
        current_q_widget = self.question_stack.currentWidget()
        if isinstance(current_q_widget, QuestionWidget):
            _, value = current_q_widget.get_value(force_validate=False)
            self.user_answers[current_q_widget.feature_name] = value

        if self.current_question_idx > 0:
            self.current_question_idx -= 1
            self._display_current_question()

    def _handle_next_or_predict_action(self):
        # Handles 'Next', 'Predict', or 'Finish' button clicks.
        current_q_widget = self.question_stack.currentWidget()

        if not isinstance(current_q_widget, QuestionWidget):
            self._emit_prediction_finished_signal()
            return

        # validate and save the answer on question widget
        is_valid, value = current_q_widget.get_value(force_validate=True)
        if not is_valid:
            return 

        self.user_answers[current_q_widget.feature_name] = value

        # Move to next question or trigger prediction
        if self.current_question_idx < len(self.questions_definitions) - 1:
            self.current_question_idx += 1
            self._display_current_question()
        else:
            # Last question answered, perform prediction
            self._initiate_model_prediction()

    def _initiate_model_prediction(self):
        #Collects all answers and asks the AlzheimerModel to predict.
        print("DEBUG: Initiating prediction. Current answers:", self.user_answers)
        # Ensure all mandatory questions are answered.
        for q_def in self.questions_definitions:
            if q_def["mandatory"] and self.user_answers.get(q_def["feature"]) is None:
                QMessageBox.warning(
                    self, "Missing Input",
                    f"The question for '{q_def['text']}' is mandatory. Please provide an answer."
                )
                # Navigate back to the missing mandatory question
                self.current_question_idx = self.questions_definitions.index(q_def)
                self._display_current_question()
                return

        try:
            prediction_input_data = self.user_answers.copy()
            print("DEBUG: Data being sent to model for prediction:", prediction_input_data)
            prediction_results = self.alzheimer_model_instance.predict(prediction_input_data)
            self._display_prediction_results(
                prediction_results['prediction'],
                prediction_results['probability']
            )
        except ValueError as ve:
             QMessageBox.critical(self, "Prediction Error", f"Input data issue: {str(ve)}")
             traceback.print_exc()
        except Exception as e:
             QMessageBox.critical(self, "Prediction Error", f"An unexpected error occurred during prediction: {str(e)}")
             traceback.print_exc()

    def _display_prediction_results(self, predicted_class: int, class_probabilities: list):
        #Displays the prediction outcome and probabilities on a new results page.
        # Clear current question widget
        while self.question_stack.count() > 0:
            widget = self.question_stack.widget(0)
            self.question_stack.removeWidget(widget)
            widget.deleteLater()

        results_page_widget = QWidget()
        results_layout = QVBoxLayout(results_page_widget)
        results_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        results_layout.setSpacing(20)
        results_layout.setContentsMargins(20, 50, 20, 50)

        result_text = "High Likelihood of Alzheimer's" if predicted_class == 1 else "Low Likelihood of Alzheimer's"
        result_color = "#FF0000" if predicted_class == 1 else "#2ECC71"

        prediction_label = QLabel(result_text)
        prediction_label.setFont(QFont("Arial", 28, QFont.Weight.Bold))
        prediction_label.setStyleSheet(f"color: {result_color}; margin-bottom: 15px;")
        prediction_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        results_layout.addWidget(prediction_label)

        # Probability Display
        prob_no_alz_text = f"Probability of No Alzheimer's: {class_probabilities[0]*100:.1f}%"
        prob_alz_text =    f"Probability of Alzheimer's:    {class_probabilities[1]*100:.1f}%"

        prob_label_style = "color: #D0D0D0; font-size: 14pt; margin-top: 5px;"
        prob_no_alz_label = QLabel(prob_no_alz_text)
        prob_no_alz_label.setStyleSheet(prob_label_style)
        prob_no_alz_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        results_layout.addWidget(prob_no_alz_label)

        prob_alz_label = QLabel(prob_alz_text)
        prob_alz_label.setStyleSheet(prob_label_style)
        prob_alz_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        results_layout.addWidget(prob_alz_label)

        # Probability Bar
        self.prediction_probability_bar = QProgressBar()
        self.prediction_probability_bar.setRange(0, 100)
        self.prediction_probability_bar.setValue(0) 
        self.prediction_probability_bar.setTextVisible(False)
 
        self.prediction_probability_bar.setMaximumWidth(600)

        results_layout.addWidget(self.prediction_probability_bar, 0, Qt.AlignmentFlag.AlignCenter)

        results_layout.addStretch(1)

        self.question_stack.addWidget(results_page_widget)
        self.question_stack.setCurrentWidget(results_page_widget)

        # Animate the progress bar
        self.animated_bar_current_value = 0
        # Target is probability of Alzheimer's (class 1)
        self.animated_bar_target_value = int(class_probabilities[1] * 100)
        self.bar_animation_timer = QTimer(self)
        self.bar_animation_timer.timeout.connect(self._run_probability_bar_animation_step)
        self.bar_animation_timer.start(15)

        self._update_navigation_buttons_state() # Update buttons for "Finish" state

    def _run_probability_bar_animation_step(self):
        #Animates the progress bar to the target probability value.
        if self.animated_bar_current_value < self.animated_bar_target_value:
            self.animated_bar_current_value += 1
            self.prediction_probability_bar.setValue(self.animated_bar_current_value)
        else:
            # Ensure it settles on the exact target value
            self.prediction_probability_bar.setValue(self.animated_bar_target_value)
            self.bar_animation_timer.stop()