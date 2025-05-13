from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QPushButton, QComboBox,
    QApplication, QMenu, QMessageBox, QStackedWidget
)
from PyQt6.QtCore import QPropertyAnimation, QEasingCurve, QRect, Qt
from PyQt6.QtGui import QFont, QMouseEvent
from model.TrainedModel import TrainedModel
from model.Model import AlzheimerModel
from .AccuracyDetailsView import AccuracyDetailsView # Already themed
from .PredictionView import PredictionView # This will be themed now
from .AddModelWindow import AddModelDialog # This will also get a style touch-up
import traceback

# --- Constants for UI elements ---
# You can adjust positions (x, y) and sizes (width, height) here
WINDOW_SIZE = (1280, 720) # Overall window size

# COMBOBOX_GEOMETRY: (x_pos, y_pos, width, height) for the model selection dropdown
COMBOBOX_GEOMETRY = (184, 50, 913, 70) # Height adjusted for new font/padding

# BUTTON_DATA: List of tuples for main screen buttons
# Each tuple: (internal_id, display_text, x_pos, y_pos, width, height, font_point_size)
BUTTON_DATA = [
    ("pred_btn", "Make a Prediction", 425, 184, 431, 94, 16),
    ("acc_btn", "View Model Details", 425, 291, 431, 94, 16),
    ("modeladd_btn", "Train a New Model", 425, 399, 431, 94, 16),
    ("exist_btn", "Exit Application", 425, 543, 431, 94, 16)
]

class CustomComboBox(QComboBox):
    """
    A custom QComboBox that populates its model list on the first click
    and provides a right-click context menu for deleting models.
    """
    def __init__(self, main_window, parent=None):
        super().__init__(parent)
        self.main_window = main_window
        self.is_populated_on_click = False # Flag to populate only once on click
        self.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.customContextMenuRequested.connect(self._show_context_menu)

    def mousePressEvent(self, event: QMouseEvent):
        # Populate the dropdown with model names the first time it's clicked
        if not self.is_populated_on_click:
            self.main_window.populate_models()
            self.is_populated_on_click = True
        super().mousePressEvent(event) # Call the base class event handler

    def _show_context_menu(self, position):
        """Displays a context menu (e.g., for deleting the selected model)."""
        current_text = self.currentText()
        # Only show menu if a real model is selected (not the placeholder)
        if current_text != "No model is currently selected" and self.currentIndex() > 0:
            context_menu = QMenu(self)
            delete_action = context_menu.addAction(f"Delete Model: '{current_text}'")
            action = context_menu.exec(self.mapToGlobal(position)) # Show menu at cursor

            if action == delete_action:
                self._confirm_and_delete_current_model()

    def _confirm_and_delete_current_model(self):
        """Asks for confirmation then deletes the currently selected model."""
        current_model_name = self.currentText()
        reply = QMessageBox.question(
            self, # Parent widget for the message box
            'Confirm Deletion',
            f'Are you absolutely sure you want to delete the model "{current_model_name}"?\n'
            'This action cannot be undone.',
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No, # Buttons
            QMessageBox.StandardButton.No # Default button
        )

        if reply == QMessageBox.StandardButton.Yes:
            try:
                was_deleted = self.main_window.model_manager.delete_model(current_model_name)
                if was_deleted:
                    self.main_window.populate_models() # Refresh the list
                    # If the deleted model was the active one, reset the selection
                    if self.main_window.current_model == current_model_name:
                          self.main_window.current_model = None
                          self.main_window.alzheimer_model = None
                          self.setCurrentText("No model is currently selected") # Reset display text
                    QMessageBox.information(self, "Success", f'Model "{current_model_name}" deleted.')
                else:
                     QMessageBox.warning(self, "Deletion Failed", f'Could not delete model "{current_model_name}".')
            except Exception as e:
                QMessageBox.critical(self, "Error", f"An error occurred while deleting the model: {str(e)}")


class MainWindow(QMainWindow):
    """
    The main application window. It handles model selection, navigation between different
    views (prediction, accuracy details, model training), and overall application flow.
    """
    def __init__(self):
        super().__init__()
        self.buttons = {} # Stores QPushButton instances, keyed by internal_id
        self.animations = {} # Stores QPropertyAnimation objects for button hover
        self.button_original_geometries = {} # Stores original QRect of buttons for animation

        self._apply_global_styles() # Apply the new theme
        self._setup_window_structure()
        self._setup_main_screen_widgets() # ComboBox and buttons on the main_widget
        # self._setup_buttons() # This is now part of _setup_main_screen_widgets

        self.model_manager = TrainedModel() # Manages saving/loading models
        self.current_model = None # Name of the currently selected model
        self.alzheimer_model = None # Instance of AlzheimerModel for the selected model

        self.populate_models() # Load available models into the ComboBox
        self.stacked_widget.setCurrentWidget(self.main_widget) # Show main screen first

    def _setup_window_structure(self):
        """Sets up the main window properties and the QStackedWidget for view management."""
        self.setWindowTitle("Alzheimer's Prediction System")
        self.setFixedSize(*WINDOW_SIZE) # Use fixed size defined in constants

        # QStackedWidget allows switching between different "pages" or views
        self.stacked_widget = QStackedWidget()
        self.setCentralWidget(self.stacked_widget)

        # Create the main screen widget (will hold ComboBox and main navigation buttons)
        self.main_widget = QWidget()
        self.stacked_widget.addWidget(self.main_widget)

        # Create the view for displaying accuracy details
        self.accuracy_view = AccuracyDetailsView(self) # Pass self for navigation back
        self.stacked_widget.addWidget(self.accuracy_view)

        # Create the view for making predictions
        self.prediction_view = PredictionView(self) # Pass self
        self.prediction_view.prediction_finished.connect(self._return_to_main_view_from_child)
        self.stacked_widget.addWidget(self.prediction_view)

    def _apply_global_styles(self):
        """Applies the modern black & white theme using QSS (Qt Style Sheets)."""
        # This QSS will style the MainWindow and its direct children that match selectors.
        # More specific styles in child widgets might override these.
        self.setStyleSheet("""
            QMainWindow {
                background-color: #1E1E1E; /* Dark grey background */
            }
            QComboBox {
                background-color: #2D2A2E; /* Slightly lighter dark grey */
                color: white;
                border: 1px solid #AAAAAA; /* Light grey border */
                border-radius: 5px;
                padding: 10px 15px; /* Vertical and horizontal padding */
                font-size: 20pt; /* Larger font for combobox */
                /* To adjust ComboBox height, padding and font-size are key.
                   The min-height can also be set if needed. */
            }
            QComboBox::drop-down { /* Arrow part of the combobox */
                border: none;
                /* You might need to add an image for a custom arrow if desired */
            }
            QComboBox QAbstractItemView { /* The dropdown list itself */
                background-color: #2D2A2E;
                color: white;
                border: 1px solid #AAAAAA;
                selection-background-color: #555555; /* Darker grey for selection */
                selection-color: white;
                outline: 0px; /* Remove focus outline on items if desired */
            }
            QPushButton { /* General style for all buttons in MainWindow context */
                background-color: #333333; /* Mid-grey */
                color: white;
                border: 1px solid #AAAAAA; /* Light grey border */
                border-radius: 8px;        /* Slightly rounded corners */
                padding: 10px;             /* Default padding */
                font-size: 16pt;           /* Default font size for main buttons */
            }
            QPushButton:hover {
                background-color: #4A4A4A; /* Lighter grey on hover */
                border: 1px solid white;   /* White border on hover */
            }
            QPushButton:disabled {
                background-color: #252525;
                color: #666666;
                border: 1px solid #444444;
            }
            QLabel { /* General style for labels if needed directly in MainWindow */
                color: white;
                /* font-size: 12pt; */ /* Example */
            }
            QMessageBox { /* Style message boxes for consistency */
                background-color: #282828; /* Dark background for message box */
                /* font-size: 11pt; */
            }
            QMessageBox QLabel { /* Text inside QMessageBox */
                color: white;
            }
            QMessageBox QPushButton { /* Buttons inside QMessageBox */
                background-color: #333333;
                color: white;
                border: 1px solid #AAAAAA;
                border-radius: 5px;
                padding: 8px 15px;
                min-width: 80px; /* Ensure buttons are not too small */
                /* font-size: 10pt; */
            }
            QMessageBox QPushButton:hover {
                background-color: #4A4A4A;
                border: 1px solid white;
            }
        """)

    def _setup_main_screen_widgets(self):
        """Sets up the ComboBox and main navigation buttons on the main_widget."""
        # --- ComboBox for Model Selection ---
        self.model_cb = CustomComboBox(self, self.main_widget)
        # To change ComboBox position and size, edit COMBOBOX_GEOMETRY constant above
        self.model_cb.setGeometry(*COMBOBOX_GEOMETRY)
        # Font is set via QSS, but can be overridden here if needed:
        # self.model_cb.setFont(QFont("Arial", 20, QFont.Weight.Bold))
        self.model_cb.setEditable(False) # User cannot type in the ComboBox
        self.model_cb.addItem("No model is currently selected") # Placeholder
        self.model_cb.currentIndexChanged.connect(self.on_model_selected)

        # --- Main Navigation Buttons ---
        for btn_id, text, x, y, width, height, font_size in BUTTON_DATA:
            button = QPushButton(text, self.main_widget)
            # To change individual button position and size, edit BUTTON_DATA constant
            button.setGeometry(x, y, width, height)
            # To change button font size, edit BUTTON_DATA or the QSS
            button.setFont(QFont("Arial", font_size))

            self.buttons[btn_id] = button
            self.button_original_geometries[button] = QRect(x, y, width, height)

            # Setup hover animation for each button
            animation = QPropertyAnimation(button, b"geometry")
            animation.setDuration(150) # Animation speed in ms
            animation.setEasingCurve(QEasingCurve.Type.InOutQuad) # Smooth animation
            self.animations[button] = animation

            button.enterEvent = lambda event, b=button: self._animate_button_hover(b, expand=True)
            button.leaveEvent = lambda event, b=button: self._animate_button_hover(b, expand=False)

            # Connect button clicks to their respective actions
            self._connect_button_action(btn_id, button)

    def _connect_button_action(self, btn_id: str, button: QPushButton):
        """Connects a button's clicked signal to the appropriate method."""
        if btn_id == "exist_btn":
            button.clicked.connect(self.close_application) # Changed from self.close
        elif btn_id == "modeladd_btn":
            button.clicked.connect(self.show_add_model_dialog)
        elif btn_id == "pred_btn":
            button.clicked.connect(self.show_prediction_view)
        elif btn_id == "acc_btn":
            button.clicked.connect(self.show_accuracy_details_view)

    def _animate_button_hover(self, button: QPushButton, expand: bool):
        """Animates button geometry on mouse hover."""
        animation = self.animations.get(button)
        if not animation: return

        if animation.state() == QPropertyAnimation.State.Running:
            animation.stop()

        original_geometry = self.button_original_geometries[button]
        current_geometry = button.geometry()

        if expand:
            scale_factor = 1.05 # How much to expand
            new_width = int(original_geometry.width() * scale_factor)
            new_height = int(original_geometry.height() * scale_factor)
            # Center the expanded button relative to its original position
            new_x = original_geometry.x() - (new_width - original_geometry.width()) // 2
            new_y = original_geometry.y() - (new_height - original_geometry.height()) // 2
            target_geometry = QRect(new_x, new_y, new_width, new_height)
        else:
            target_geometry = original_geometry # Revert to original

        animation.setStartValue(current_geometry)
        animation.setEndValue(target_geometry)
        animation.start()

    def populate_models(self):
        """Fills the ComboBox with names of available trained models."""
        current_selection = self.model_cb.currentText() # Preserve selection if possible
        self.model_cb.blockSignals(True) # Avoid triggering on_model_selected multiple times
        self.model_cb.clear()
        self.model_cb.addItem("No model is currently selected")
        models = self.model_manager.list_models()
        if models:
            self.model_cb.addItems(models)
            # Try to restore previous selection
            if current_selection in models:
                self.model_cb.setCurrentText(current_selection)
            elif self.current_model and self.current_model in models: # if model was loaded programmatically
                self.model_cb.setCurrentText(self.current_model)

        self.model_cb.blockSignals(False)


    def on_model_selected(self, index: int):
        """Handles logic when a model is selected from the ComboBox."""
        selected_model_name = self.model_cb.currentText()

        if selected_model_name != "No model is currently selected" and index > 0:
            if self.current_model == selected_model_name and self.alzheimer_model is not None:
                print(f"Model '{selected_model_name}' is already loaded.")
                return # Avoid reloading if already selected and loaded

            self.current_model = selected_model_name
            print(f"Attempting to load model: '{self.current_model}'...")
            try:
                # This is where the model file is read and preprocessors are initialized
                self.alzheimer_model = AlzheimerModel(model_name=self.current_model)
                if self.alzheimer_model.model is None: # Check if the core ML model loaded
                     QMessageBox.critical(
                         self, "Load Error",
                         f"Failed to load components for model '{self.current_model}'. "
                         "The model file might be corrupted or incompatible."
                     )
                     self._reset_model_selection()
                else:
                    print(f"Model '{self.current_model}' loaded successfully.")
            except Exception as e:
                print(f"ERROR loading model '{selected_model_name}': {e}")
                traceback.print_exc()
                QMessageBox.critical(self, "Error", f"Error loading model '{selected_model_name}':\n{str(e)}")
                self._reset_model_selection()
        else:
            # "No model is currently selected" or index 0 was chosen
            self._reset_model_selection()
            print("No model selected or selection cleared.")

    def _reset_model_selection(self):
        """Resets current model and updates ComboBox if necessary."""
        self.current_model = None
        self.alzheimer_model = None
        if self.model_cb.currentIndex() != 0: # If combobox isn't already on placeholder
            self.model_cb.setCurrentIndex(0)


    def show_add_model_dialog(self):
        """
        Opens a dialog for the user to enter a name for a new model.
        If a valid name is provided and confirmed, it proceeds to train and save the new model.
        """
        print("DEBUG: Opening 'Add New Model' dialog...")
        # Create an instance of the AddModelDialog, parented to this MainWindow
        add_model_dialog = AddModelDialog(self)

        if add_model_dialog.exec():
            # Dialog was accepted, get the model name entered by the user
            model_name_to_train = add_model_dialog.getModelName()
            print(f"DEBUG: Dialog accepted. Model name to train: '{model_name_to_train}'")
            # Ensure a valid model name was actually returned (though AddModelDialog should validate emptiness)
            if model_name_to_train:
                # Optional: Check if a model with this name already exists to prevent overwriting
                # without explicit confirmation (this logic would typically be here or in TrainedModel)
                existing_models = self.model_manager.list_models()
                if model_name_to_train in existing_models:
                    reply = QMessageBox.question(
                        self,
                        "Model Exists",
                        f"A model named '{model_name_to_train}' already exists.\n"
                        "Do you want to overwrite it?",
                        QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                        QMessageBox.StandardButton.No
                    )
                    if reply == QMessageBox.StandardButton.No:
                        print(f"DEBUG: User chose not to overwrite existing model '{model_name_to_train}'.")
                        return # Exit the method without training
                print(f"DEBUG: Proceeding to train model: '{model_name_to_train}'")
                try:
                    training_model_instance = AlzheimerModel()
                    print("DEBUG: AlzheimerModel instance created for training.")

                    training_results = training_model_instance.train(model_name_to_train)
                    print(f"DEBUG: Training complete for '{model_name_to_train}'. Results obtained.")

                    # Display a success message to the user
                    QMessageBox.information(
                        self,
                        "Training Successful",
                        f"Model '{model_name_to_train}' has been trained and saved successfully!\n\n"
                        f"Test Accuracy: {training_results.get('test_accuracy', 'N/A'):.4f}\n"
                        f"Mean CV Score (on training data): {training_results.get('cv_data', {}).get('mean_cv_score', 'N/A'):.4f}"
                    )
                    # Refresh the list of models in the ComboBox
                    print("DEBUG: Repopulating models in ComboBox...")
                    self.populate_models()
                    # Automatically select the newly trained model in the ComboBox
                    self.model_cb.setCurrentText(model_name_to_train)
                except Exception as e:
                    # Catch any error during the training or saving process
                    error_message = f"Failed to train model '{model_name_to_train}'.\n\nError: {str(e)}"
                    print(f"ERROR: Exception during training/saving for '{model_name_to_train}':")
                    traceback.print_exc() # Print full traceback to console for debugging
                    QMessageBox.critical(self, "Training Error", error_message)
            else:
                print("DEBUG: Model name was empty after dialog was accepted (this should be prevented by dialog validation).")
        else:
            print("DEBUG: 'Add New Model' dialog was cancelled by the user.")


    def show_prediction_view(self):
        """Switches to the PredictionView if a model is loaded."""
        if self.alzheimer_model is None or self.alzheimer_model.model is None:
            QMessageBox.warning(self, "No Model Loaded", "Please select and load a valid model first.")
            return

        self.prediction_view.set_model_and_start(self.alzheimer_model)
        self.stacked_widget.setCurrentWidget(self.prediction_view)

    def show_accuracy_details_view(self):
        """Switches to the AccuracyDetailsView if a model is selected."""
        if self.current_model is None:
            QMessageBox.warning(self, "No Model Selected", "Please select a model to view its details.")
            return
        self.accuracy_view.set_model_for_details(self.current_model) # Pass model name
        self.stacked_widget.setCurrentWidget(self.accuracy_view)

    def _return_to_main_view_from_child(self):
        """Slot to switch back to the main_widget, called by child views."""
        # Potentially clear state in the child view if needed before switching
        if self.stacked_widget.currentWidget() == self.prediction_view:
            self.prediction_view.reset_view_state() # Add a method to PredictionView to clear answers etc.

        self.stacked_widget.setCurrentWidget(self.main_widget)

    def close_application(self):
        """Closes the application."""
        QApplication.quit()