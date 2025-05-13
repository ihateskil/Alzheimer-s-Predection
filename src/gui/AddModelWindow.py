
from PyQt6.QtWidgets import QDialog, QVBoxLayout, QLineEdit, QPushButton, QMessageBox, QLabel, QHBoxLayout
from PyQt6.QtGui import QFont

class AddModelDialog(QDialog):
    """Dialog for getting a new model name from the user before training."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.entered_model_name = "" # Store the validated name
        self._setup_ui()

    def _setup_ui(self):
        self.setWindowTitle("Train New Model")
        self.setFixedSize(450, 200)
        self._setup_layout_and_widgets()
        self._apply_dialog_styles()

    def _setup_layout_and_widgets(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(15)
        layout.setContentsMargins(20, 20, 20, 20) # Padding inside the dialog

        # Instruction Label
        instruction_label = QLabel("Enter a unique name for the new model:")
        instruction_label.setFont(QFont("Arial", 11))
        instruction_label.setWordWrap(True)
        layout.addWidget(instruction_label)

        # Text Input for Model Name
        self.name_input_field = QLineEdit()
        self.name_input_field.setPlaceholderText("e.g., MyAlzheimerModel_v1")
        self.name_input_field.setFont(QFont("Arial", 12))
        layout.addWidget(self.name_input_field)

        # layout for buttons
        button_layout = QHBoxLayout()
        button_layout.setSpacing(10)

        self.confirm_train_button = QPushButton("Confirm and Train")
        self.confirm_train_button.clicked.connect(self._validate_and_accept)
        button_layout.addWidget(self.confirm_train_button)

        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.clicked.connect(self.reject) # Closes dialog, returns False from exec()
        button_layout.addWidget(self.cancel_button)

        layout.addLayout(button_layout)

    def _apply_dialog_styles(self):
        """Apply stylesheets to the dialog and its components."""
        self.setStyleSheet("""
            QDialog {
                background-color: #282828; /* Dark background for the dialog itself */
                border: 1px solid #555555; /* Optional border for the dialog window */
            }
            QLabel {
                color: #E0E0E0; /* Off-white for labels within this dialog */
                margin-bottom: 5px; /* Ensure consistent margin for instruction_label */
            }
            QLineEdit {
                background-color: #1E1E1E; /* Darker input background, consistent with MainWindow */
                color: white;
                border: 1px solid #AAAAAA;
                border-radius: 4px;
                padding: 8px 10px;
                min-height: 28px; 
                font-size: 12pt;
            }
            QLineEdit::placeholder {
                color: #777777; /* Grey for placeholder text */
            }
            QPushButton {
                background-color: #383838; /* Buttons slightly different from main window for context */
                color: white;
                border: 1px solid #777777;
                border-radius: 4px;
                padding: 8px 15px;
                font-size: 11pt;
                min-height: 28px;
            }
            QPushButton:hover {
                background-color: #4A4A4A; /* Consistent hover effect */
                border: 1px solid #CCCCCC;
            }
            QPushButton:disabled { /* If you ever disable buttons in this dialog */
                background-color: #252525;
                color: #666666;
            }
        """)

    def _validate_and_accept(self):
        #Validates the entered model name and accepts the dialog if valid
        model_name_candidate = self.name_input_field.text().strip()

        if not model_name_candidate:
            QMessageBox.warning(self, "Input Required", "Model name cannot be empty.")
            return

        invalid_chars = " /\\:*?\"<>|"
        if any(char in model_name_candidate for char in invalid_chars):
             QMessageBox.warning(self, "Invalid Name",
                                 f"Model name contains invalid characters.\n"
                                 f"Avoid: {invalid_chars}")
             return

        self.entered_model_name = model_name_candidate
        self.accept()

    def getModelName(self) -> str:
        """Returns the validated model name entered by the user."""
        return self.entered_model_name

