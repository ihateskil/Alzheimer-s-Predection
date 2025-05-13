from PyQt6.QtWidgets import (
    QWidget, QPushButton, QDialog, QTabWidget, QVBoxLayout,
    QLabel, QScrollArea, QMessageBox
)
from PyQt6.QtGui import QFont
from PyQt6.QtCore import Qt, QTimer, QPropertyAnimation, QEasingCurve, QRect
import pyqtgraph as pg
import numpy as np
from model.ModelAccuracy import ModelAccuracy
import colorcet

class VisualizationDialog(QDialog):
    """
    A dialog window for displaying detailed model performance metrics using various plots and text.
    It uses a QTabWidget to organize different metrics into separate tabs with a modern B&W theme.
    """
    def __init__(self, model_name: str, metrics: dict, parent: QWidget = None):
        super().__init__(parent)
        self.setWindowTitle(f"Model Performance Details: {model_name}")
        self.setMinimumSize(750, 700)
        self.metrics = metrics
        self.model_name = model_name

        self._setup_dialog_ui()
        self._apply_dialog_theme()

        self.plot_animation_timer = QTimer(self)
        self.plot_animation_timer.timeout.connect(self._run_plot_animations_step)
        self.plot_animation_timer.start(40)
        self.animation_progress_value = 0.0

    def _setup_dialog_ui(self):
        dialog_main_layout = QVBoxLayout(self)
        self.metrics_tabs_widget = QTabWidget()
        dialog_main_layout.addWidget(self.metrics_tabs_widget)

        # --- Test Accuracy & CV Scores Summary ---
        test_accuracy_tab_page = QWidget()
        test_accuracy_layout = QVBoxLayout(test_accuracy_tab_page)
        self.test_accuracy_glw = pg.GraphicsLayoutWidget()
        self.test_accuracy_plot_item = self.test_accuracy_glw.addPlot()
        self.test_accuracy_bar_graph = pg.BarGraphItem(x=[0], height=[0], width=0.4, brush='#CCCCCC')
        self.test_accuracy_plot_item.addItem(self.test_accuracy_bar_graph)
        self.test_accuracy_plot_item.setYRange(0, 1)
        self.test_accuracy_plot_item.setXRange(-0.5, 0.5)
        test_accuracy_layout.addWidget(self.test_accuracy_glw)

        test_accuracy_summary_text = QLabel()
        test_accuracy_summary_text.setFont(QFont("Arial", 11))
        test_accuracy_summary_text.setTextFormat(Qt.TextFormat.RichText)
        test_accuracy_summary_text.setText(
            f"<b>Test Set Accuracy:</b> {self.metrics.get('test_accuracy', 0.0):.4f}<br>"
            f"<b>Mean Cross-Validation Score (Training):</b> {self.metrics.get('cv_scores', {}).get('mean', 0.0):.4f}<br>"
            f"<b>Std Dev of CV Scores:</b> {self.metrics.get('cv_scores', {}).get('std', 0.0):.4f}"
        )
        test_accuracy_layout.addWidget(test_accuracy_summary_text)
        self.metrics_tabs_widget.addTab(test_accuracy_tab_page, "Accuracy Overview")

        # --- Detailed Classification Report ---
        class_report_tab_page = QScrollArea()
        class_report_tab_page.setWidgetResizable(True)
        class_report_content_widget = QWidget()
        class_report_main_layout = QVBoxLayout(class_report_content_widget)
        class_report_display_text = QLabel()
        class_report_display_text.setFont(QFont("Arial", 11))
        class_report_display_text.setTextFormat(Qt.TextFormat.RichText)
        class_report_display_text.setWordWrap(True)
        report_data = self.metrics.get('classification_report', {})
        report_html_str = "<h3>Classification Report (Test Set):</h3>"
        for class_label, stats in report_data.items():
            if class_label in ['0', '1']:
                class_name_str = "No Alzheimer's (Class 0)" if class_label == '0' else "Alzheimer's (Class 1)"
                report_html_str += f"<p><b>{class_name_str}:</b><br>"
                report_html_str += f"  Precision: {stats.get('precision', 0.0):.4f}<br>"
                report_html_str += f"  Recall: {stats.get('recall', 0.0):.4f}<br>"
                report_html_str += f"  F1-Score: {stats.get('f1-score', 0.0):.4f}<br>"
                report_html_str += f"  Support: {stats.get('support', 0)}</p>"
        report_html_str += f"<p><b>Overall Accuracy:</b> {report_data.get('accuracy', 0.0):.4f}<br>"
        report_html_str += f"<b>Macro Average F1-Score:</b> {report_data.get('macro avg', {}).get('f1-score', 0.0):.4f}<br>"
        report_html_str += f"<b>Weighted Average F1-Score:</b> {report_data.get('weighted avg', {}).get('f1-score', 0.0):.4f}</p>"
        class_report_display_text.setText(report_html_str)
        class_report_main_layout.addWidget(class_report_display_text)
        class_report_tab_page.setWidget(class_report_content_widget)
        self.metrics_tabs_widget.addTab(class_report_tab_page, "Classification Report")

        # --- Confusion Matrix Visualization ---
        confusion_matrix_tab_page = QWidget()
        cm_layout = QVBoxLayout(confusion_matrix_tab_page)
        self.cm_glw = pg.GraphicsLayoutWidget() # Store GLW
        self.confusion_matrix_plot_item = self.cm_glw.addPlot()
        self.confusion_matrix_heatmap_image = pg.ImageItem(np.zeros((2, 2)))
        try:
            selected_cmap = None
            if colorcet:
                try: selected_cmap = pg.colormap.get('CET-L1')
                except ValueError: print("Info: 'CET-L1' colormap not found, trying 'gray'.")
            if not selected_cmap:
                try: selected_cmap = pg.colormap.get('gray')
                except ValueError: print("Info: 'gray' colormap not found. Creating manual grayscale.")
            if not selected_cmap:
                colors = [(0,0,0),(255,255,255)]; positions=np.linspace(0,1,len(colors))
                selected_cmap = pg.ColorMap(positions, np.array(colors, dtype=np.uint8))
        except Exception as e_cmap_load:
            print(f"Critical Error setting colormap: {e_cmap_load}. Defaulting to basic manual grayscale.")
            colors = [(0,0,0),(255,255,255)]; positions=np.linspace(0,1,len(colors))
            selected_cmap = pg.ColorMap(positions, np.array(colors, dtype=np.uint8))
        self.confusion_matrix_heatmap_image.setLookupTable(selected_cmap.getLookupTable())
        self.confusion_matrix_plot_item.addItem(self.confusion_matrix_heatmap_image)
        self.confusion_matrix_plot_item.setAspectLocked(True)
        self.confusion_matrix_plot_item.getAxis('bottom').setTicks([[(0, 'Predicted Negative'), (1, 'Predicted Positive')]])
        self.confusion_matrix_plot_item.getAxis('left').setTicks([[(0, 'Actual Negative'), (1, 'Actual Positive')]])
        cm_layout.addWidget(self.cm_glw)
        confusion_matrix_values_text = QLabel()
        confusion_matrix_values_text.setFont(QFont("Arial", 10))
        cm_array_data = np.array(self.metrics.get('confusion_matrix', [[0,0],[0,0]]))
        tn, fp, fn, tp = cm_array_data.ravel() if cm_array_data.size == 4 else (0,0,0,0)
        confusion_matrix_values_text.setText(
            f"<b>True Negatives (TN):</b> {tn}\n<b>False Positives (FP):</b> {fp}\n"
            f"<b>False Negatives (FN):</b> {fn}\n<b>True Positives (TP):</b> {tp}"
        )
        confusion_matrix_values_text.setTextFormat(Qt.TextFormat.RichText)
        cm_layout.addWidget(confusion_matrix_values_text)
        self.metrics_tabs_widget.addTab(confusion_matrix_tab_page, "Confusion Matrix")

        # --- Feature Importance Scores ---
        feature_importance_tab_page = QScrollArea()
        feature_importance_tab_page.setWidgetResizable(True)
        feat_imp_content_widget = QWidget()
        feat_imp_layout = QVBoxLayout(feat_imp_content_widget)
        feat_imp_display_text = QLabel()
        feat_imp_display_text.setFont(QFont("Arial", 10))
        feat_imp_display_text.setWordWrap(True)
        importance_scores = self.metrics.get('feature_importance', [])
        selected_feature_names = self.metrics.get('selected_columns', [])
        if selected_feature_names and importance_scores:
            sorted_features = sorted(zip(selected_feature_names, importance_scores), key=lambda item: item[1], reverse=True)
            feat_imp_str_content = "<h3>Selected Feature Importances (Mutual Information):</h3><ul>"
            for col_name, importance_val in sorted_features:
                feat_imp_str_content += f"<li><b>{col_name}:</b> {importance_val:.4f}</li>"
            feat_imp_str_content += "</ul>"
        else:
            feat_imp_str_content = "<p>Feature importance scores are not available.</p>"
        feat_imp_display_text.setTextFormat(Qt.TextFormat.RichText)
        feat_imp_display_text.setText(feat_imp_str_content)
        feat_imp_layout.addWidget(feat_imp_display_text)
        feature_importance_tab_page.setWidget(feat_imp_content_widget)
        self.metrics_tabs_widget.addTab(feature_importance_tab_page, "Feature Importance")

        # --- Cross-Validation Scores from Training ---
        cv_scores_tab_page = QWidget()
        cv_layout = QVBoxLayout(cv_scores_tab_page)
        self.cv_scores_glw = pg.GraphicsLayoutWidget() # Store GLW
        self.cv_scores_plot_item = self.cv_scores_glw.addPlot()
        cv_scores_values = self.metrics.get('cv_scores', {}).get('scores', [0.0]*5)
        num_cv_folds = len(cv_scores_values)
        self.cv_scores_bar_graph = pg.BarGraphItem(x=np.arange(num_cv_folds), height=[0]*num_cv_folds, width=0.3, brush='#CCCCCC')
        self.cv_scores_plot_item.addItem(self.cv_scores_bar_graph)
        self.cv_scores_plot_item.setYRange(0, 1)
        self.cv_scores_plot_item.setXRange(-0.5, num_cv_folds - 0.5)
        self.cv_scores_plot_item.getAxis('bottom').setTicks([[(i, f'Fold {i+1}') for i in range(num_cv_folds)]])
        cv_layout.addWidget(self.cv_scores_glw)
        cv_summary_text = QLabel()
        cv_summary_text.setFont(QFont("Arial", 10))
        cv_mean_score = self.metrics.get('cv_scores', {}).get('mean', 0.0)
        cv_std_dev_score = self.metrics.get('cv_scores', {}).get('std', 0.0)
        cv_scores_list_str = "<br>".join([f"  Fold {i+1}: {score:.4f}" for i, score in enumerate(cv_scores_values)])
        cv_summary_text.setTextFormat(Qt.TextFormat.RichText)
        cv_summary_text.setText(
            f"<b>Cross-Validation Scores (training):</b><br>{cv_scores_list_str}<br>"
            f"<b>Mean CV Accuracy:</b> {cv_mean_score:.4f}<br><b>Std Dev:</b> {cv_std_dev_score:.4f}"
        )
        cv_layout.addWidget(cv_summary_text)
        self.metrics_tabs_widget.addTab(cv_scores_tab_page, "Cross-Validation")

        # --- Model Learning Curve ---
        learning_curve_tab_page = QWidget()
        lc_layout = QVBoxLayout(learning_curve_tab_page)
        self.learning_curve_glw = pg.GraphicsLayoutWidget()
        lc_layout.addWidget(self.learning_curve_glw)
        learning_curve_data = self.metrics.get('learning_curve')
        self.learning_curve_plot_item = None # Initialize, will be created if data exists

        if learning_curve_data and learning_curve_data.get('train_sizes_abs'):
            train_sample_sizes = np.array(learning_curve_data['train_sizes_abs'])
            train_mean_scores = np.array(learning_curve_data.get('train_scores_mean', []))
            train_std_scores = np.array(learning_curve_data.get('train_scores_std', []))
            valid_mean_scores = np.array(learning_curve_data.get('valid_scores_mean', []))
            valid_std_scores = np.array(learning_curve_data.get('valid_scores_std', []))

            if all(arr.size > 0 for arr in [train_sample_sizes, train_mean_scores, train_std_scores, valid_mean_scores, valid_std_scores]):
                self.learning_curve_plot_item = self.learning_curve_glw.addPlot() # Create PlotItem here
                self.learning_curve_plot_item.addLegend(offset=(-30, 30))
                self.learning_curve_plot_item.plot(train_sample_sizes, valid_mean_scores, pen=pg.mkPen('white', width=2), name='Validation Score')
                self.learning_curve_plot_item.plot(train_sample_sizes, train_mean_scores, pen=pg.mkPen('#AAAAAA', width=2), name='Training Score')
                valid_score_fill = pg.FillBetweenItem(
                    pg.PlotCurveItem(train_sample_sizes, valid_mean_scores + valid_std_scores),
                    pg.PlotCurveItem(train_sample_sizes, valid_mean_scores - valid_std_scores),
                    brush=pg.mkBrush(200,200,200,70)
                )
                train_score_fill = pg.FillBetweenItem(
                    pg.PlotCurveItem(train_sample_sizes, train_mean_scores + train_std_scores),
                    pg.PlotCurveItem(train_sample_sizes, train_mean_scores - train_std_scores),
                    brush=pg.mkBrush(150,150,150,70)
                )
                self.learning_curve_plot_item.addItem(valid_score_fill)
                self.learning_curve_plot_item.addItem(train_score_fill)
                self.learning_curve_plot_item.setLabel('bottom', 'Training Samples')
                self.learning_curve_plot_item.setLabel('left', 'Accuracy Score')
                self.learning_curve_plot_item.setYRange(0, 1.05)
            else:
                lc_layout.addWidget(QLabel("Learning curve data incomplete."))
        else:
             lc_layout.addWidget(QLabel("Learning curve data unavailable."))
        self.metrics_tabs_widget.addTab(learning_curve_tab_page, "Learning Curve")

        model_info_label = QLabel(f"Metrics for model: <b>{self.model_name}</b> (Naive Bayes)")
        model_info_label.setFont(QFont("Arial", 11))
        model_info_label.setTextFormat(Qt.TextFormat.RichText)
        model_info_label.setStyleSheet("margin-top: 10px;")
        dialog_main_layout.addWidget(model_info_label, alignment=Qt.AlignmentFlag.AlignCenter)

    def _run_plot_animations_step(self):
        self.animation_progress_value = min(self.animation_progress_value + 0.05, 1.0)
        if self.animation_progress_value >= 1.0:
            self.plot_animation_timer.stop()

        test_acc_val = self.metrics.get('test_accuracy', 0.0)
        if hasattr(self, 'test_accuracy_bar_graph'): # Check if item exists
            self.test_accuracy_bar_graph.setOpts(height=[test_acc_val * self.animation_progress_value])

        cm_data_array = np.array(self.metrics.get('confusion_matrix', [[0,0],[0,0]]))
        max_cm_val = cm_data_array.max() if cm_data_array.size > 0 else 1.0
        if hasattr(self, 'confusion_matrix_heatmap_image'): # Check if item exists
            self.confusion_matrix_heatmap_image.setImage(
                cm_data_array * self.animation_progress_value,
                levels=(0, max_cm_val * self.animation_progress_value + 1e-6)
            )

        cv_scores_data = self.metrics.get('cv_scores', {}).get('scores', [])
        if hasattr(self, 'cv_scores_bar_graph'): # Check if item exists
            self.cv_scores_bar_graph.setOpts(height=[val * self.animation_progress_value for val in cv_scores_data])

    def _apply_dialog_theme(self):
        self.setStyleSheet("background-color: #1E1E1E; color: #E0E0E0;")
        self.metrics_tabs_widget.setStyleSheet("""
            QTabWidget::pane {
                border: 1px solid #444444; border-top: 2px solid #555555;
                background-color: #282828;
            }
            QTabWidget::tab-bar { alignment: left; }
            QTabBar::tab {
                background: #333333; color: #E0E0E0; border: 1px solid #444444;
                border-bottom: none; border-top-left-radius: 5px; border-top-right-radius: 5px;
                min-width: 120px; padding: 10px 15px; margin-right: 1px;
            }
            QTabBar::tab:selected {
                background: #282828; color: white; border-bottom: 2px solid #282828;
            }
            QTabBar::tab:!selected:hover { background: #4A4A4A; }
        """)

        # Style the plot items directly after they are created.
        if hasattr(self, 'test_accuracy_plot_item'):
            self.test_accuracy_glw.setBackground('#282828')
            self._apply_theme_to_single_plotitem(self.test_accuracy_plot_item, "Test Accuracy Overview")
        if hasattr(self, 'confusion_matrix_plot_item'):
            self.cm_glw.setBackground('#282828')
            self._apply_theme_to_single_plotitem(self.confusion_matrix_plot_item, "Confusion Matrix")
        if hasattr(self, 'cv_scores_plot_item'):
            self.cv_scores_glw.setBackground('#282828')
            self._apply_theme_to_single_plotitem(self.cv_scores_plot_item, "Cross-Validation Scores")
        if hasattr(self, 'learning_curve_plot_item') and self.learning_curve_plot_item: # Check if it was created
            self.learning_curve_glw.setBackground('#282828')
            self._apply_theme_to_single_plotitem(self.learning_curve_plot_item, "Model Learning Curve")


    def _apply_theme_to_single_plotitem(self, plot_item_to_style: pg.PlotItem, title: str = ""):
        plot_view_box = plot_item_to_style.getViewBox()
        axis_pen_style = pg.mkPen(color='#AAAAAA', width=1)
        plot_item_to_style.getAxis('bottom').setPen(axis_pen_style)
        plot_item_to_style.getAxis('left').setPen(axis_pen_style)
        plot_item_to_style.getAxis('bottom').setTextPen('#E0E0E0')
        plot_item_to_style.getAxis('left').setTextPen('#E0E0E0')

        if plot_view_box:
            plot_view_box.setBorder(pen=pg.mkPen('#666666', width=1))

        # Set or update title using the passed argument
        plot_item_to_style.setTitle(title, color='#E0E0E0', size='12pt')

        plot_legend = plot_item_to_style.legend
        if plot_legend:
            plot_legend.setLabelTextColor('#E0E0E0')


    def tabWidget(self) -> QTabWidget:
        return self.metrics_tabs_widget


class AccuracyDetailsView(QWidget):
    def __init__(self, main_window: QWidget):
        super().__init__()
        self.main_window_ref = main_window
        self.active_model_name = None
        self.metric_buttons = {}
        self.button_animations = {}
        self.button_base_geometries = {}
        self._setup_accuracy_view_ui()

    def _setup_accuracy_view_ui(self):
        action_button_definitions = [
            ("View Accuracy Overview", 200,184,431,94,16, lambda: self._trigger_visualization_dialog(0)),
            ("Show Classification Report", 200,291,431,94,16, lambda: self._trigger_visualization_dialog(1)),
            ("Display Confusion Matrix", 200,398,431,94,16, lambda: self._trigger_visualization_dialog(2)),
            ("Analyze Feature Importance", 649,184,431,94,16, lambda: self._trigger_visualization_dialog(3)),
            ("Examine CV Scores", 649,291,431,94,16, lambda: self._trigger_visualization_dialog(4)),
            ("Plot Training Curve", 649,398,431,94,16, lambda: self._trigger_visualization_dialog(5)),
            ("Return to Main Menu", 425,540,431,94,16, self._navigate_back_to_main_menu)
        ]
        for btn_text, x, y, w, h, fs, cb in action_button_definitions:
            btn = QPushButton(btn_text, self)
            btn.setGeometry(x,y,w,h); btn.setFont(QFont("Arial",fs)); btn.clicked.connect(cb)
            self.metric_buttons[btn_text] = btn
            self.button_base_geometries[btn] = QRect(x,y,w,h)
            anim = QPropertyAnimation(btn, b"geometry"); anim.setDuration(150)
            anim.setEasingCurve(QEasingCurve.Type.InOutQuad)
            self.button_animations[btn] = anim
            btn.enterEvent = lambda e, b=btn: self._handle_button_hover_animation(b, True)
            btn.leaveEvent = lambda e, b=btn: self._handle_button_hover_animation(b, False)
        self.setStyleSheet("""
            QWidget { background-color: #1E1E1E; }
            QPushButton {
                background-color:#333333; color:white; border:1px solid #AAAAAA;
                border-radius:8px; padding:10px; font-size:16pt;
            }
            QPushButton:hover { background-color:#4A4A4A; border:1px solid white; }
            QPushButton:disabled { background-color:#252525; color:#666666; border:1px solid #444444; }
        """)

    def _handle_button_hover_animation(self, btn:QPushButton, expand:bool):
        anim = self.button_animations.get(btn);
        if not anim: return
        if anim.state() == QPropertyAnimation.State.Running: anim.stop()
        orig_geom = self.button_base_geometries[btn]; curr_geom = btn.geometry()
        if expand:
            sf=1.05; nw=int(orig_geom.width()*sf); nh=int(orig_geom.height()*sf)
            nx=orig_geom.x()-(nw-orig_geom.width())//2; ny=orig_geom.y()-(nh-orig_geom.height())//2
            tgt_geom = QRect(nx,ny,nw,nh)
        else: tgt_geom = orig_geom
        anim.setStartValue(curr_geom); anim.setEndValue(tgt_geom); anim.start()

    def set_model_for_details(self, model_name: str):
        self.active_model_name = model_name

    def _trigger_visualization_dialog(self, tab_index: int = 0):
        if not self.active_model_name:
            QMessageBox.warning(self,"No Model Selected","Select a model before viewing details.")
            return
        try:
            model_acc_info = ModelAccuracy(self.active_model_name)
            metrics = model_acc_info.compute_metrics()
            if tab_index==5 and (not metrics.get('learning_curve') or not metrics['learning_curve'].get('train_sizes_abs')):
                QMessageBox.information(self,"Learning Curve Data Missing",
                                        "Learning curve data unavailable for this model.\nRetrain to generate.")
                return
            dialog = VisualizationDialog(self.active_model_name, metrics, self)
            if 0 <= tab_index < dialog.tabWidget().count():
                dialog.tabWidget().setCurrentIndex(tab_index)
            dialog.exec()
        except ValueError as ve:
            QMessageBox.critical(self, "Metric Error",f"Could not load metrics for '{self.active_model_name}':\n{ve}")
        except Exception as e:
            QMessageBox.critical(self, "Unexpected Error",f"Error preparing model details:\n{e}")

    def _navigate_back_to_main_menu(self):
        if hasattr(self.main_window_ref, 'stacked_widget') and hasattr(self.main_window_ref, 'main_widget'):
            self.main_window_ref.stacked_widget.setCurrentWidget(self.main_window_ref.main_widget)
        else: print("Error: MainWindow components not found for navigation.")