import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import math
from matplotlib.ticker import FormatStrFormatter

# Load Excel file
file_path = r"E:\Desktop\Thesis\Actual tensile strength $ -Final\data split\Dataset_33_Rows.xlsx"
df = pd.read_excel(file_path)

# Define predictors and targets
predictors = ["fiber_width", "fiber_factor"]
targets = ["A.I. fracture strain", "A.I. failure stress", "A.I. failure strain", "A.I. fracture length", "A.I. Young’s Modulus", "fiber_factor", "fiber_branch"]

# Ensure predictors and targets exist in the dataset
missing_predictors = [col for col in predictors if col not in df.columns]
missing_targets = [col for col in targets if col not in df.columns]

if missing_predictors:
    print(f"Error: Missing predictor columns: {missing_predictors}")
    exit()
if missing_targets:
    print(f"Error: Missing target columns: {missing_targets}")
    exit()

# Drop NaN values if any
if df.isna().sum().sum() > 0:
    print("Warning: Data contains NaN values. Dropping rows with NaNs.")
    df = df.dropna()

X = df[predictors]
y = df[targets]

# Split data (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the Random Forest model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Predict on test set
y_pred = rf_model.predict(X_test)

# Calculate evaluation metrics
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r_squared = r2_score(y_test, y_pred)

# Print overall results
print("Random Forest Regression Evaluation:")
print(f"Mean Absolute Error (MAE): {mae:.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
print(f"R-squared (R²): {r_squared:.4f}")

# Plot observed vs predicted values
plt.figure(figsize=(12, 8))
for i, target in enumerate(targets):
    plt.subplot(3, 4, i+1)
    sns.scatterplot(x=y_test[target].values, y=y_pred[:, i], alpha=0.7, color="blue", label="Predicted")
    plt.plot([y_test[target].min(), y_test[target].max()],
             [y_test[target].min(), y_test[target].max()],
             'r--', label=f"Perfect Fit (R²={r2_score(y_test[target], y_pred[:, i]):.4f})")
    plt.xlabel("Observed Values")
    plt.ylabel("Predicted Values")
    plt.title(f"{target}")
    plt.legend()
    plt.grid(True)

plt.tight_layout()
plt.show()
