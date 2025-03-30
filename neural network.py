import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import random
import math
from matplotlib.ticker import FormatStrFormatter

#  Load the dataset
file_path = r"E:\Desktop\Thesis\Actual tensile strength $ -Final\Dataset.xlsx"
df = pd.read_excel(file_path)

#  Define predictors and targets
predictors = ["fiber_width","fiber_factor", "fiber_branch"]
targets = ["A.I. Fracture Stress","A.I. Fracture Strain", "A.I. Failure Stress",
           "A.I. Failure Strain", "A.I. Young's Modulus","A.I. Toughness"]

# Ensure columns exist
missing_columns = [col for col in predictors + targets if col not in df.columns]
if missing_columns:
    print(f"Error: Missing columns - {missing_columns}")
    exit()

#  Prepare data
X = df[predictors].values
y = df[targets].values

# Handle missing values
df.dropna(inplace=True)
X = df[predictors].values
y = df[targets].values

#  Split data (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#  Standardize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

#  Build Neural Network Model
model = keras.Sequential([
    keras.layers.Dense(32, activation='relu', input_shape=(X_train.shape[1],)),  # Input Layer
    keras.layers.Dense(64, activation='relu'),  # Hidden Layer 1
    keras.layers.Dense(32, activation='relu'),  # Hidden Layer 2
    keras.layers.Dense(y_train.shape[1])  # Output Layer
])



#  Compile model
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

#  Train model
history = model.fit(X_train, y_train, epochs=500, batch_size=8, validation_data=(X_test, y_test), verbose=1)

#  Predict on train and test data
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

#  Evaluate Performance Separately for Train & Test
train_mae = mean_absolute_error(y_train, y_train_pred)
test_mae = mean_absolute_error(y_test, y_test_pred)
train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
train_r2 = r2_score(y_train, y_train_pred)
test_r2 = r2_score(y_test, y_test_pred)

#  Print Overall Evaluation
print("\nNeural Network Regression Evaluation:")
print(f"Train MAE: {train_mae:.4f}, Test MAE: {test_mae:.4f}")
print(f"Train RMSE: {train_rmse:.4f}, Test RMSE: {test_rmse:.4f}")
print(f"Train R²: {train_r2:.4f}, Test R²: {test_r2:.4f}")

#  Save Predictions to Excel
results_df = pd.DataFrame({
    "Observed_" + target: y_test[:, i] for i, target in enumerate(targets)
})
for i, target in enumerate(targets):
    results_df["Predicted_" + target] = y_test_pred[:, i]

excel_path = r"E:\Desktop\Thesis\Actual tensile strength $ -Final\Neural_Network_Results.xlsx"
results_df.to_excel(excel_path, index=False)
print(f"Data with predictions saved to {excel_path}")

#  Determine rows and columns dynamically
num_targets = len(targets)
cols = 3  # 3 columns for better readability
rows = math.ceil(num_targets / cols)  # Dynamically calculate rows

#  Increase figure size to improve spacing
plt.figure(figsize=(cols * 6, rows * 6))  # Scale figure size dynamically

for i, target in enumerate(targets):
    ax = plt.subplot(rows, cols, i + 1)  # Dynamically adjust subplot grid

    # Calculate individual R² score
    r2_target_test = r2_score(y_test[:, i], y_test_pred[:, i])

    # Scatter plot with larger markers
    sns.scatterplot(x=y_test[:, i], y=y_test_pred[:, i], s=50, color="blue", label="Predicted", ax=ax)

    # Y = X trendline
    ax.plot([y_test[:, i].min(), y_test[:, i].max()],
            [y_test[:, i].min(), y_test[:, i].max()],
            'r--', label="Perfect Fit")

    ax.set_xlabel("Observed Values", fontsize=14)
    ax.set_ylabel("Predicted Values", fontsize=14)
    ax.set_title(f"{target}\nR²: {r2_target_test:.4f}", fontsize=16)
    ax.legend()
    ax.grid(True)

    #  Ensure consistent decimal formatting for both axes
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))  # 2 decimal places for X-axis
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))  # 2 decimal places for Y-axis

    #  Ensure equal aspect ratio
    ax.set_aspect('equal', adjustable='datalim')

#  Increase spacing between plots
plt.subplots_adjust(hspace=0.6, wspace=0.5)

#  Save as high-quality image
plt.savefig(r"E:\Desktop\Thesis\Actual tensile strength $ -Final\Observed_vs_Predicted.png", dpi=300, bbox_inches="tight")

plt.show()

