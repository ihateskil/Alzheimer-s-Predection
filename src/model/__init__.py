import tkinter as tk
from model import AlzheimerModel
from gui import AlzheimerGUI

def test_model():
    model = AlzheimerModel()
    model.train()

def run_gui():
    root = tk.Tk()
    app = AlzheimerGUI(root)
    root.geometry("400x600")
    root.mainloop()

if __name__ == "__main__":
    print("Choose an option:")
    print("1. Test model accuracy")
    print("2. Run GUI")
    choice = input("Enter your choice (1 or 2): ")

    if choice == "1":
        test_model()
    elif choice == "2":
        run_gui()
    else:
        print("Invalid choice. Please run again and select 1 or 2.")