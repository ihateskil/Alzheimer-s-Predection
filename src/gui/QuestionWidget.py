from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit,
    QRadioButton, QButtonGroup, QSizePolicy, QMessageBox
)
from PyQt6.QtGui import QFont, QDoubleValidator, QIntValidator
from PyQt6.QtCore import Qt

class QuestionWidget(QWidget):
    # reusable widget to display a single question and provide an appropriate

    def __init__(self, feature_name: str, question_text: str, input_type: str,
                 options: dict = None, data_range: tuple = None, is_mandatory: bool = False,
                 hint_text: str = None, unit_text: str = None, parent=None):
        super().__init__(parent)
        self.feature_name = feature_name
        self.question_text = question_text
        self.input_type = input_type
        self.options = options or {}
        self.data_range = data_range
        self.is_mandatory_input = is_mandatory
        self.hint_text = hint_text
        self.unit_text = unit_text

        self.current_answer_value = None # Stores the validated answer
        self._init_ui_structure()
        self._apply_widget_styles() 

    def _init_ui_structure(self):
        # Sets up the layout and interactive elements for the question
        self.main_layout = QVBoxLayout(self)
        self.main_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.main_layout.setContentsMargins(20, 20, 20, 20)
        self.main_layout.setSpacing(20)

        # --- Question Label ---
        self.question_display_label = QLabel(self.question_text)
        self.question_display_label.setFont(QFont("Arial", 28, QFont.Weight.Bold)) # Adjusted for better fit
        self.question_display_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.question_display_label.setWordWrap(True)
        self.question_display_label.setMaximumWidth(600)
        self.question_display_label.setMinimumHeight(200)
        self.main_layout.addWidget(self.question_display_label, 0, Qt.AlignmentFlag.AlignCenter)

        if self.hint_text:
            self.hint_display_label = QLabel(f"({self.hint_text})")
            self.hint_display_label.setFont(QFont("Arial", 12, italic=True))
            self.hint_display_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.hint_display_label.setWordWrap(True)
            self.hint_display_label.setMaximumWidth(800)
            self.hint_display_label.setMaximumHeight(400)
            self.main_layout.addWidget(self.hint_display_label, 0, Qt.AlignmentFlag.AlignCenter)


        # --- Input Area ---
        input_elements_container = QWidget()
        input_elements_container.setMaximumWidth(800)
        input_elements_container.setMaximumHeight(200)
        self.input_elements_layout = QVBoxLayout(input_elements_container)
        input_elements_container.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Maximum)

        if self.input_type == 'text':
            self._create_text_input_field()
        elif self.input_type in ['binary_radio', 'categorical_radio']:
            self._create_radio_button_group()
        else:
            error_label = QLabel(f"Error: Unknown input type '{self.input_type}' for question.")
            self.input_elements_layout.addWidget(error_label)

        self.main_layout.addWidget(input_elements_container, 0, Qt.AlignmentFlag.AlignCenter)
        self.main_layout.addStretch(1)

    def _apply_widget_styles(self):
       # Applies specific QSS styling to elements within this QuestionWidget.
        self.setStyleSheet("""
            /* QLabel styling within QuestionWidget context */
            QuestionWidget > QLabel { /* Targets direct QLabel children like question_display_label */
                color: #E0E0E0; /* Off-white for question text */
            }
            /* Hint label specifically if needed, can be targeted by object name */
            QLabel#hintLabel { /* Assuming you set objectName="hintLabel" */
                color: #A0A0A0; /* Lighter grey for hint */
            }

            QLineEdit {
                background-color: #282828;
                color: white;
                border: 1px solid #AAAAAA;
                border-radius: 5px;
                /* INCREASED PADDING AND FONT SIZE FOR INPUT TEXT */
                padding: 14px 18px; /* Increased padding */
                min-height: 42px;   /* Increased min-height */
                font-size: 16pt;    /* INCREASED font size for text inside QLineEdit */
                /* To control QLineEdit width more directly: */
                min-width: 500px;   /* Example: Set a minimum width */
                max-width: 500px;   /* Example: Set a maximum width */
            }
            QLineEdit::placeholder {
                color: #777777; /* Placeholder text color */
            }

            QRadioButton {
                color: #D0D0D0; /* Radio button text color */
                spacing: 10px;   /* Space between indicator and text */
                padding: 8px;
                font-size: 13pt; /* Font size for radio button labels */
            }
            QRadioButton::indicator {
                width: 18px;     /* Size of the radio button circle */
                height: 18px;
            }
            QRadioButton::indicator::unchecked {
                border: 2px solid #AAAAAA;
                border-radius: 10px; /* Make it circular */
                background-color: #282828; /* Background when unchecked */
            }
            QRadioButton::indicator::checked {
                border: 2px solid #6CA0DC;  /* Accent color border when checked */
                border-radius: 10px;
                background-color: #6CA0DC; /* Accent color fill when checked */
            }
            QRadioButton::indicator::unchecked:hover {
                border: 2px solid #CCCCCC;
            }
             QRadioButton::indicator::checked:hover {
                border: 2px solid #8CBEEE;
            }
        """)
        if hasattr(self, 'hint_display_label'): self.hint_display_label.setObjectName("hintLabel")


    def _create_text_input_field(self):
        self.line_edit_input = QLineEdit()
        # Font, min-height, padding are now primarily controlled by QSS (_apply_widget_styles)
        # self.line_edit_input.setFont(QFont("Arial", 14))

        placeholder = "Enter your answer"
        if self.data_range:
            range_str = f"({self.data_range[0]} - {self.data_range[1]})"
            placeholder = f"Enter value {range_str}"
            if self.unit_text:
                placeholder += f" in {self.unit_text}"

            # Validator setup
            is_integer_range = all(isinstance(x, int) or (isinstance(x, float) and x.is_integer()) for x in self.data_range)
            if is_integer_range:
                validator = QIntValidator(int(self.data_range[0]), int(self.data_range[1]), self)
            else:
                # Allow for a reasonable number of decimal places
                validator = QDoubleValidator(float(self.data_range[0]), float(self.data_range[1]), 2, self)
                validator.setNotation(QDoubleValidator.Notation.StandardNotation) # Avoid scientific notation
            self.line_edit_input.setValidator(validator)

        self.line_edit_input.setPlaceholderText(placeholder)
        self.line_edit_input.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.input_elements_layout.addWidget(self.line_edit_input, 0, Qt.AlignmentFlag.AlignCenter)

  
        if self.unit_text and not self.data_range:
            unit_label = QLabel(self.unit_text)
            unit_label.setFont(QFont("Arial", 10))
            unit_label.setStyleSheet("color: #A0A0A0; margin-left: 5px;")
            self.input_elements_layout.addWidget(unit_label, 0, Qt.AlignmentFlag.AlignCenter)


    def _create_radio_button_group(self):
        self.radio_button_selection_group = QButtonGroup(self)
        self.radio_button_selection_group.setExclusive(True)

        # Use QHBoxLayout for binary (2 options) for side-by-side, QVBoxLayout for more.
        if self.input_type == 'binary_radio' and len(self.options) == 2:
            radio_layout = QHBoxLayout()
            radio_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
            radio_layout.setSpacing(30)
        else: # Categorical or more than 2 binary options
            radio_layout = QVBoxLayout()
            radio_layout.setAlignment(Qt.AlignmentFlag.AlignCenter) # Center items vertically
            radio_layout.setSpacing(10)

        for display_text, actual_value in self.options.items():
            radio_btn = QRadioButton(display_text)
            radio_btn.setProperty("actual_value", actual_value) # Store the backend value
            self.radio_button_selection_group.addButton(radio_btn)
            if isinstance(radio_layout, QHBoxLayout):
                radio_layout.addWidget(radio_btn)
            else: # 
                radio_layout.addWidget(radio_btn, 0, Qt.AlignmentFlag.AlignHCenter)

        self.input_elements_layout.addLayout(radio_layout)

    def get_value(self, force_validate: bool = True):
        """
        Retrieves and validates the user's answer from the input field.
        If force_validate is False, it returns the current input without showing error popups
        """
        self.current_answer_value = None
        error_msg_text = None

        if self.input_type == 'text':
            entered_text = self.line_edit_input.text().strip()
            if not entered_text: 
                if self.is_mandatory_input and force_validate:
                    error_msg_text = f"This field is mandatory. Please enter a value for '{self.question_text.split('?')[0]}'."
                else:
                    # If not mandatory or not validating, None is acceptable for empty
                    return True, None
            else:
                if self.line_edit_input.hasAcceptableInput():
                    # Convert to appropriate number type
                    try:
                        # Handle comma as decimal separator if locale uses it, then convert to float
                        val_text_for_conversion = entered_text.replace(',', '.')
                        is_integer_range = self.data_range and all(isinstance(x, int) or (isinstance(x, float) and x.is_integer()) for x in self.data_range)

                        if is_integer_range:
                            self.current_answer_value = int(float(val_text_for_conversion))
                        else: 
                            self.current_answer_value = float(val_text_for_conversion)

                        # Double-check range explicitly, as validator might allow intermediate states
                        if self.data_range and not (self.data_range[0] <= self.current_answer_value <= self.data_range[1]):
                            error_msg_text = f"Value for '{self.question_text.split('?')[0]}' must be between {self.data_range[0]} and {self.data_range[1]}."
                            self.current_answer_value = None
                    except ValueError:
                        error_msg_text = f"Invalid number format for '{self.question_text.split('?')[0]}'."
                        self.current_answer_value = None

                elif force_validate:
                    error_msg_text = f"Invalid input for '{self.question_text.split('?')[0]}'."
                    if self.data_range:
                        error_msg_text += f" Expected range: {self.data_range[0]}-{self.data_range[1]}."
                    if self.unit_text:
                         error_msg_text += f" (Unit: {self.unit_text})."


        elif self.input_type in ['binary_radio', 'categorical_radio']:
            selected_button = self.radio_button_selection_group.checkedButton()
            if selected_button:
                self.current_answer_value = selected_button.property("actual_value")
            elif self.is_mandatory_input and force_validate:
                error_msg_text = f"This field is mandatory. Please select an option for '{self.question_text.split('?')[0]}'."

        if error_msg_text and force_validate:
            QMessageBox.warning(self, "Input Error", error_msg_text)
            return False, None 

        return True, self.current_answer_value # Validation passed or not forced

    def set_current_answer(self, value_to_set):
        """Pre-fills the input field with a given value (e.g., when navigating back)."""
        if self.input_type == 'text' and value_to_set is not None:
            self.line_edit_input.setText(str(value_to_set))
        elif self.input_type in ['binary_radio', 'categorical_radio'] and value_to_set is not None:
            for button in self.radio_button_selection_group.buttons():
                if button.property("actual_value") == value_to_set:
                    button.setChecked(True)
                    break
        self.current_answer_value = value_to_set # Store it as if user entered it