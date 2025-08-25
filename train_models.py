# import required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.feature_selection import RFE, SelectFromModel
from sklearn.metrics import mean_absolute_error , mean_squared_error, r2_score
import seaborn as sns
from tqdm import tqdm
import joblib
import json
import os
import warnings
warnings.filterwarnings("ignore")


class HousePriceNet(nn.Module):
    def __init__(self, input_dim):
        super(HousePriceNet, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.model(x)


#-------------------------------TASK 1: LOAD & CLEAN DATA------------------------------#

# Read the housing dataset
df = pd.read_csv('house_data.csv')
print(df.head())

# View statistical summary
print(df.describe())

# Drop `id` and `date` columns. These features will not be used for analysis
df = df.drop(['id', 'date'], axis=1)

# Data dimensionality
print(f'Shape: {df.shape}')   # Shape: (21613, 19)

# Check to see all null values
print(df.isna().sum())

# Summary of dataframe
print(df.info())


#-----------------------------TASK 2: DATA VISUALIZATION------------------------------#

# Column names
cols = list(df.columns)

# Determine the number of rows and columns for the subplots
n_features = len(cols)
n_cols = 4 
n_rows = (n_features + n_cols - 1) // n_cols

# Create the subplots
fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4, n_rows * 2))
axes = axes.flatten()

for i, col in enumerate(cols):
    sns.histplot(data=df, x=col, kde=True, ax=axes[i])
    plt.title(f'Distribution of {col}')

# Remove any unused subplots
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.show()


# Correlaxtion matrix
corr = df.corr()

# Plot matrix using Seaborn heatmap
plt.figure(figsize=(12, 10))
ax = sns.heatmap(corr, annot=True, cmap='Blues', fmt=".3f")
ax.xaxis.tick_top()
ax.tick_params(axis='x', labelrotation=45)
ax.tick_params(axis='y', labelrotation=45)
plt.xticks(fontsize=7)
plt.yticks(fontsize=7)
plt.show()


#-----------------------------TASK 3: DATA PREPROCESSING-----------------------------#

# Seperate the data into input features (X) and target variable (y)
X = df.drop('price', axis=1)
y = df['price']

# Split data into train and test dataset (20%)
x_train , x_test , y_train , y_test = train_test_split(X, y, random_state=100, test_size=0.2)
print(f'X_train shape: {x_train.shape},  X_test shape: {x_test.shape}')
print(f'y_train shape: {y_train.shape},  y_test shape: {y_test.shape}')

features = list(x_train.columns)
target = 'price'

#-------------------------------TASK 4: DEFINE MODELS-------------------------------#

# Initialize and train models
models = {
    "Linear Regression": {
        'model': LinearRegression(),
        'params': {} # none for this model
    },
    "Ridge Regression": {
        'model': DecisionTreeRegressor(),
        "params": {
            "max_depth": [3, 5, 20, 10, None],
            "min_samples_split": [2, 5, 20, 10]
        }
    },
    "Random Forest": {
        "model": RandomForestRegressor(),
        "params": {
            "n_estimators": [50, 100],
            "max_depth": [5, 10, None],
            "min_samples_split": [2, 5]
        }
    },
    "Svm": {
        "model": SVR(),
        "params": {
            "C": [0.1, 1, 10],
            "kernel": ["linear", "rbf"],
            "gamma": ["scale", "auto"]
        }
    },
    "Xgboost": {
        # tree_method='gpu_hist': Uses GPU-accelerated histogram-based algorithm.
        # predictor='gpu_predictor': Ensures prediction also runs on GPU.

        "model": XGBRegressor(tree_method='gpu_hist', predictor='gpu_predictor', verbosity=0),
        "params": {
            "n_estimators": [100, 200],
            "max_depth": [3, 6],
            "learning_rate": [0.05, 0.1],
            "subsample": [0.7, 1.0]
        }
    },
    # "Neural Network": {
    #     "model": MLPRegressor(max_iter=2000, early_stopping=True, validation_fraction=0.1),
    #     "params": {
    #         "hidden_layer_sizes": [(64,), (128,), (64, 64)],
    #         "activation": ["relu", "tanh"],
    #         "learning_rate_init": [0.001, 0.005, 0.01]
    #     }
    # }
}


#------------------------------TASK 5: FEATURE SELECTION------------------------------#

def select_features(name, model, x_tr, y_tr, x_te):
    selected_features = features
    x_tr_fs, x_te_fs = x_tr, x_te

    if name == "Linear Regression":
        rfe = RFE(estimator=model, n_features_to_select=10)
        rfe.fit(x_tr, y_tr)
        x_tr_fs = rfe.transform(x_tr)
        x_te_fs = rfe.transform(x_te)
        selected_features = [f for f, s in zip(features, rfe.support_) if s]

    elif name in ["Random Forest", "Xgboost"]:
        model.fit(x_tr, y_tr)
        sfm = SelectFromModel(model, threshold="median")
        sfm.fit(x_tr, y_tr)
        x_tr_fs = sfm.transform(x_tr)
        x_te_fs = sfm.transform(x_te)
        selected_features = [f for f, s in zip(features, sfm.get_support()) if s]

    return x_tr_fs, x_te_fs, selected_features


#------------------------TASK 6: MODEL TRAINING & EVALUATION--------------------------#

model_dir = "models"  # Ensure this folder exists
features_dir = "features"
scaler_dir = "scalers"
results = {}
selected_feature_sets = {}

# Create the folder if it doesn't exist
os.makedirs(model_dir, exist_ok=True)
os.makedirs(features_dir, exist_ok=True)
os.makedirs(scaler_dir, exist_ok=True)

# Remove all files in the folder
for filename in os.listdir(model_dir):
    file_path = os.path.join(model_dir, filename)
    try:
        if os.path.isfile(file_path):
            os.remove(file_path)
    except Exception as e:
        print(f"Error deleting file {file_path}: {e}")

for filename in os.listdir(features_dir):
    file_path = os.path.join(features_dir, filename)
    try:
        if os.path.isfile(file_path):
            os.remove(file_path)
    except Exception as e:
        print(f"Error deleting file {file_path}: {e}")

for filename in os.listdir(scaler_dir):
    file_path = os.path.join(scaler_dir, filename)
    try:
        if os.path.isfile(file_path):
            os.remove(file_path)
    except Exception as e:
        print(f"Error deleting file {file_path}: {e}")


for name, config in tqdm(models.items(), desc="Training Models", unit="model"):
    print(f"\nTraining {name}...")

    # Feature selection
    x_train_fs, x_test_fs, selected = select_features(name, config['model'], x_train, y_train, x_test)
    selected_feature_sets[name] = selected

    # Initialize and fit the scaler after feature selection
    scaler = StandardScaler()
    scaler.fit(x_train_fs)

    # Transform both training and test data
    x_train_fs_scaled = scaler.transform(x_train_fs)
    x_test_fs_scaled = scaler.transform(x_test_fs)

    # Train with grid search
    grid = GridSearchCV(config["model"], config["params"], cv=3, scoring='neg_mean_squared_error', n_jobs=-1)
    grid.fit(x_train_fs_scaled, y_train)
    best_model = grid.best_estimator_

    # Save model and scaler
    joblib.dump(best_model, f"{model_dir}/{name.replace(' ', '_')}.pkl")
    joblib.dump(scaler, f"{scaler_dir}/{name.replace(' ', '_')}_scaler.pkl")


    # Prediction
    y_pred = best_model.predict(x_test_fs_scaled)

    results[name] = {
        "Best Parameters": grid.best_params_,
        "MAE": mean_absolute_error(y_test, y_pred),
        "MSE": mean_squared_error(y_test, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_test, y_pred)),
        "R2 Score": r2_score(y_test, y_pred),
        "Selected Features": selected
    }

#----------------------TASK 6B: TRAIN NEURAL NETWORK VIA GPU--------------------------#

print("\nTraining Neural Network...")

name = "Neural Network"

# Use all features for PyTorch model (no feature selection)
selected_feature_sets[name] = features

# Scale input features
scaler_nn = StandardScaler()
x_train_scaled_nn = scaler_nn.fit_transform(x_train)
x_test_scaled_nn = scaler_nn.transform(x_test)

# Scale target variable
y_train_scaled = scaler_nn.fit_transform(y_train.values.reshape(-1, 1))
y_test_scaled = scaler_nn.transform(y_test.values.reshape(-1, 1))

# Save scaler
joblib.dump(scaler_nn, f"{scaler_dir}/{name.replace(' ', '_')}_scaler.pkl")

# Convert to PyTorch tensors
x_train_tensor = torch.tensor(x_train_scaled_nn, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train_scaled, dtype=torch.float32)
x_test_tensor = torch.tensor(x_test_scaled_nn, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test_scaled, dtype=torch.float32)

# Create DataLoader
train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize model
model_nn = HousePriceNet(x_train_tensor.shape[1]).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model_nn.parameters(), lr=0.001)

# Training loop
for epoch in range(100):
    model_nn.train()
    running_loss = 0.0
    for batch_x, batch_y in train_loader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        optimizer.zero_grad()
        outputs = model_nn(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {running_loss:.4f}")

# Evaluation
model_nn.eval()
with torch.no_grad():
    predictions_scaled = model_nn(x_test_tensor.to(device)).cpu().numpy()
    predictions = scaler_nn.inverse_transform(predictions_scaled)

    mae = mean_absolute_error(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, predictions)

# Append results
results[name] = {
    "Best Parameters": {"hidden_layer_sizes": [128, 64], "activation": "ReLU", "optimizer": "Adam", "lr": 0.001},
    "MAE": mae,
    "MSE": mse,
    "RMSE": rmse,
    "R2 Score": r2,
    "Selected Features": features
}


# Display results at the end
for model_name, res in results.items():
    print(f"\n{model_name}:")
    for k, v in res.items():
        print(f"{k}: {v}")


# Save feature names
with open(f"{features_dir}/all_features.json", "w") as f:
    json.dump(features, f)

# Save selected feature names
with open(f"{features_dir}/selected_features.json", "w") as f:
    json.dump(selected_feature_sets, f)

# Save model names
with open(f"{model_dir}/model_names.json", "w") as f:
    json.dump(list(models.keys()), f)


# Find the model with the highest R² Score
best_model_name = max(results, key=lambda name: results[name]["R2 Score"])
best_model_result = results[best_model_name]

print("\n" + "="*50)
print(f"Best Performing Model: {best_model_name}")
print("="*50)
print(f"R² Score: {best_model_result['R2 Score']:.4f}")
print(f"RMSE: {best_model_result['RMSE']:.2f}")
print(f"MAE: {best_model_result['MAE']:.2f}")
print(f"Best Parameters: {best_model_result['Best Parameters']}")
print(f"Selected Features ({len(best_model_result['Selected Features'])}):")
for feat in best_model_result['Selected Features']:
    print(f" - {feat}")
print("="*50)

