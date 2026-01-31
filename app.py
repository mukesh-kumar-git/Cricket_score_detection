import pandas as pd
import numpy as np
import gradio as gr
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# Load dataset (Hugging Face uses relative paths)
df = pd.read_csv("cricket_preprocessed.csv")

# Features and target
X = df[['overs', 'runs', 'wickets', 'balls_remaining', 'run_rate']]
y = df['final_score']

# Train-test split
Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Train model
model = RandomForestRegressor(n_estimators=200, random_state=42)
model.fit(Xtrain, ytrain)

# Evaluate model
ypred = model.predict(Xtest)
r2 = r2_score(ytest, ypred)
n = Xtest.shape[0]
p = Xtest.shape[1]
adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)
mae = mean_absolute_error(ytest, ypred)
mse = mean_squared_error(ytest, ypred)

# Prediction function for Gradio
def predict_score(overs, runs, wickets, balls_remaining, run_rate):
    input_data = np.array([[overs, runs, wickets, balls_remaining, run_rate]])
    pred = model.predict(input_data)[0]

    return (
        int(pred),
        f"{r2 * 100:.2f}%",
        f"{adj_r2 * 100:.2f}%",
        f"{mae:.2f}",
        f"{mse:.2f}"
    )

# Gradio Interface
interface = gr.Interface(
    fn=predict_score,
    inputs=[
        gr.Number(label="Overs Completed"),
        gr.Number(label="Runs Scored"),
        gr.Number(label="Wickets Fallen"),
        gr.Number(label="Balls Remaining"),
        gr.Number(label="Current Run Rate")
    ],
    outputs=[
        gr.Number(label="Predicted Final Score"),
        gr.Textbox(label="R² Score"),
        gr.Textbox(label="Adjusted R²"),
        gr.Textbox(label="MAE"),
        gr.Textbox(label="MSE")
    ],
    title="🏏 T20 Cricket Score Prediction",
    description="Predict final score using Random Forest on cleaned T20 dataset"
)

# Launch app
interface.launch()
