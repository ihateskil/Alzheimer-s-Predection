import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd
from model import AlzheimerModel

class AlzheimerGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Alzheimer's Prediction")
        self.model = AlzheimerModel()
        self.model.train()  # Train the model on startup

        # Features expected by the model (excluding PatientID and Diagnosis)
        self.features = [
            'Age', 'Gender', 'Ethnicity', 'EducationLevel', 'BMI', 'Smoking',
            'AlcoholConsumption', 'PhysicalActivity', 'DietQuality', 'SleepQuality',
            'FamilyHistoryAlzheimers', 'CardiovascularDisease', 'Diabetes', 'Depression',
            'HeadInjury', 'Hypertension', 'SystolicBP', 'DiastolicBP', 'CholesterolTotal',
            'CholesterolLDL', 'CholesterolHDL', 'CholesterolTriglycerides', 'MMSE',
            'FunctionalAssessment', 'MemoryComplaints', 'BehavioralProblems', 'ADL',
            'Confusion', 'Disorientation', 'PersonalityChanges', 'DifficultyCompletingTasks',
            'Forgetfulness'
        ]

        # Create a canvas and scrollbar for the form
        self.canvas = tk.Canvas(root)
        self.scrollbar = ttk.Scrollbar(root, orient="vertical", command=self.canvas.yview)
        self.scrollable_frame = ttk.Frame(self.canvas)

        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        )

        self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.canvas.configure(yscrollcommand=self.scrollbar.set)

        self.canvas.pack(side="left", fill="both", expand=True)
        self.scrollbar.pack(side="right", fill="y")

        # Dictionary to store entry widgets
        self.entries = {}

        # Create input fields for each feature
        for idx, feature in enumerate(self.features):
            label = ttk.Label(self.scrollable_frame, text=feature + ":")
            label.grid(row=idx, column=0, padx=5, pady=2, sticky="e")
            
            entry = ttk.Entry(self.scrollable_frame)
            entry.grid(row=idx, column=1, padx=5, pady=2)
            self.entries[feature] = entry

        # Predict button
        predict_button = ttk.Button(self.scrollable_frame, text="Predict", command=self.predict)
        predict_button.grid(row=len(self.features), column=0, columnspan=2, pady=10)

        # Result label
        self.result_label = ttk.Label(self.scrollable_frame, text="Prediction: N/A")
        self.result_label.grid(row=len(self.features) + 1, column=0, columnspan=2, pady=5)

    def predict(self):
        try:
            # Collect user inputs
            input_data = {}
            for feature in self.features:
                value = self.entries[feature].get()
                if value == "":
                    raise ValueError(f"Missing value for {feature}")
                input_data[feature] = float(value)

            # Make prediction
            result = self.model.predict(input_data)

            # Display result
            prediction = "No Alzheimer's" if result['prediction'] == 0 else "Alzheimer's"
            prob_no = result['probability'][0] * 100
            prob_yes = result['probability'][1] * 100
            result_text = (f"Prediction: {prediction}\n"
                          f"Probability of No Alzheimer's: {prob_no:.2f}%\n"
                          f"Probability of Alzheimer's: {prob_yes:.2f}%")
            self.result_label.config(text=result_text)

        except ValueError as e:
            messagebox.showerror("Input Error", str(e))
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {str(e)}")

if __name__ == "__main__":
    root = tk.Tk()
    app = AlzheimerGUI(root)
    root.geometry("400x600")  # Set window size
    root.mainloop()