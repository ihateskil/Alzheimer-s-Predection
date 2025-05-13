"""GUI components for the Alzheimer's Prediction application."""

from .MainWindow import MainWindow
from .AddModelWindow import AddModelDialog
from .AccuracyDetailsView import AccuracyDetailsView
from .PredictionView import PredictionView

__all__ = [
    'MainWindow',
    'AddModelDialog',
    'AccuracyDetailsView',
    'PredictionView'
]