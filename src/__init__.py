from model import Model
import pandas as pd

# Train the model
model = Model()
model.train()

# Simulate new user input data (replace with actual data)
new_data = pd.DataFrame({
    col: [0.5] for col in model.selected_columns  # Replace with actual values
})

# Predict Alzheimer's diagnosis
predictions = model.predict(new_data)
print("Prediction (0 = Negative, 1 = Positive):", predictions)