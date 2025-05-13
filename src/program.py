import sys
import os

def run_app():
    from PyQt6.QtWidgets import QApplication
    from gui.MainWindow import MainWindow

    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    src_path = os.path.dirname(os.path.abspath(__file__))
    if src_path not in sys.path:
        sys.path.append(src_path)

    run_app()