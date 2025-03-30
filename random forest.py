import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import math
from matplotlib.ticker import FormatStrFormatter

# Load Excel file
file_path = r"E:\Desktop\Thesis\Actual tensile strength $ -Final\data split\Dataset_33_Rows.xlsx"
df = pd.read_excel(file_path)

# Define predictors and targets
predictors = ["fiber_width","fiber_factor", "fiber_branch"]
targets = [
"A.I. Fracture Stress","A.I. Fracture Strain", "A.I. Failure Stress",
           "A.I. Failure Strain", "A.I. Young's Modulus","A.I. Toughness"
]
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

# Define feature (X) and target (y)
X = df[predictors]
y = df[targets]

# Split data (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define hyperparameter grid
param_grid = {
    'n_estimators': [50, 100, 200, 500],  # Number of trees
    'max_depth': [None, 5, 10, 20, 30],  # Tree depth
    'min_samples_split': [2, 5, 10],  # Min samples for a split
    'min_samples_leaf': [1, 2, 4]  # Min samples at leaf
}

# Initialize Random Forest model
rf_model = RandomForestRegressor(random_state=42)

# Perform Grid Search with Cross Validation
grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid,
                           scoring='neg_mean_squared_error', cv=3, verbose=1, n_jobs=-1)

# Fit the model on training data
grid_search.fit(X_train, y_train)

# Get the best estimator from GridSearchCV
best_rf_model = grid_search.best_estimator_

# Make predictions on train and test sets
y_train_pred = best_rf_model.predict(X_train)
y_test_pred = best_rf_model.predict(X_test)

# Print best hyperparameters
print("\nBest Hyperparameters Found:", grid_search.best_params_)


# Function to evaluate model performance
def evaluate_model(y_true, y_pred, dataset_type="Test"):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r_squared = r2_score(y_true, y_pred)

    print(f"\n{dataset_type} Evaluation:")
    print(f"MAE: {mae:.4f}, RMSE: {rmse:.4f}, R²: {r_squared:.4f}")

    return mae, rmse, r_squared


# Evaluate train and test errors
train_mae, train_rmse, train_r2 = evaluate_model(y_train, y_train_pred, "Train")
test_mae, test_rmse, test_r2 = evaluate_model(y_test, y_test_pred, "Test")

# Store evaluation metrics per target
metrics = {"Target": [], "Train MAE": [], "Test MAE": [],
           "Train RMSE": [], "Test RMSE": [], "Train R²": [], "Test R²": []}

for i, target in enumerate(targets):
    train_mae_t = mean_absolute_error(y_train[target], y_train_pred[:, i])
    test_mae_t = mean_absolute_error(y_test[target], y_test_pred[:, i])
    train_rmse_t = np.sqrt(mean_squared_error(y_train[target], y_train_pred[:, i]))
    test_rmse_t = np.sqrt(mean_squared_error(y_test[target], y_test_pred[:, i]))
    train_r2_t = r2_score(y_train[target], y_train_pred[:, i])
    test_r2_t = r2_score(y_test[target], y_test_pred[:, i])

    metrics["Target"].append(target)
    metrics["Train MAE"].append(train_mae_t)
    metrics["Test MAE"].append(test_mae_t)
    metrics["Train RMSE"].append(train_rmse_t)
    metrics["Test RMSE"].append(test_rmse_t)
    metrics["Train R²"].append(train_r2_t)
    metrics["Test R²"].append(test_r2_t)

    print(f"{target}: Train R²={train_r2_t:.4f}, Test R²={test_r2_t:.4f}")

# Save Predictions and Metrics to Excel
output_df = pd.DataFrame({
    "Observed_" + target: y_test[target].values.flatten() for target in targets
})

for i, target in enumerate(targets):
    output_df["Predicted_" + target] = y_test_pred[:, i].flatten()  # Ensure 1D array

output_path = r"E:\Desktop\Thesis\Actual tensile strength $ -Final\Random_Forest_Regression_Results.xlsx"
with pd.ExcelWriter(output_path) as writer:
    output_df.to_excel(writer, sheet_name="Predictions", index=False)
    pd.DataFrame(metrics).to_excel(writer, sheet_name="Metrics", index=False)

print(f"\nPredictions and evaluation metrics saved to: {output_path}")
# Feature Importance
feature_importances = best_rf_model.feature_importances_
feature_importance_df = pd.DataFrame({'Feature': predictors, 'Importance': feature_importances})
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)



# Plot Feature Importance
plt.figure(figsize=(10, 6))
sns.barplot(x="Importance", y="Feature", data=feature_importance_df, palette="viridis", legend=False)
plt.xlabel("Feature Importance")
plt.ylabel("Feature")
plt.title("Feature Importance in Random Forest Model")
plt.show()

print("\nFeature Importance:")
print(feature_importance_df)

# Visualization: Observed vs Predicted Values
num_targets = len(targets)
cols = 3  # 3 columns for better readability
rows = math.ceil(num_targets / cols)  # Dynamically calculate rows

plt.figure(figsize=(cols * 6, rows * 6))  # Scale figure size dynamically

for i, target in enumerate(targets):
    ax = plt.subplot(rows, cols, i + 1)  # Dynamically adjust subplot grid

    # Calculate individual R² score
    r2_target_test = r2_score(y_test.iloc[:, i], y_test_pred[:, i])

    # Scatter plot with larger markers
    sns.scatterplot(x=y_test.iloc[:, i], y=y_test_pred[:, i], s=50, color="blue", label="Predicted", ax=ax)

    # Y = X trendline
    ax.plot(
        [y_test.iloc[:, i].min(), y_test.iloc[:, i].max()],
        [y_test.iloc[:, i].min(), y_test.iloc[:, i].max()],
        'r--', label="Perfect Fit"
    )

    ax.set_xlabel("Observed Values", fontsize=14)
    ax.set_ylabel("Predicted Values", fontsize=14)
    ax.set_title(f"{target}\nR²: {r2_target_test:.4f}", fontsize=16)
    ax.legend()
    ax.grid(True)

    # Ensure consistent decimal formatting for both axes
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))  # 2 decimal places for X-axis
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))  # 2 decimal places for Y-axis

    # Ensure equal aspect ratio
    ax.set_aspect('equal', adjustable='datalim')

# Increase spacing between plots
plt.subplots_adjust(hspace=0.6, wspace=0.5)

# Save as high-quality image
plt.savefig(r"E:\Desktop\Thesis\Actual tensile strength $ -Final\Observed_vs_Predicted.png", dpi=300, bbox_inches="tight")

plt.show()
