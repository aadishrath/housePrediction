# 🏡 House Price Prediction

This project builds and evaluates multiple machine learning models to predict house prices based on various features. It includes data preprocessing, feature selection, hyperparameter tuning, and model evaluation. The best-performing model is saved for future use.

- **Data preprocessing**: cleaning, scaling, and feature selection
- **Model training**: using six different regression algorithms
- **Hyperparameter tuning**: via `GridSearchCV` with cross-validation
- **Evaluation**: using MAE, MSE, RMSE, and R² Score
- **Model saving**: best models, scalers, and selected features are saved for future use

### 🔍 Models Used

The following models are trained and tuned:

- **Linear Regression**: baseline model with no hyperparameters
- **Decision Tree Regressor** (labeled as "Ridge Regression"):
  - `max_depth`, `min_samples_split`
- **Random Forest Regressor**:
  - `n_estimators`, `max_depth`, `min_samples_split`
- **Support Vector Machine (SVR)**:
  - `C`, `kernel`, `gamma`
- **XGBoost Regressor**:
  - `n_estimators`, `max_depth`, `learning_rate`, `subsample`
- **Neural Network (MLPRegressor)**:
  - `hidden_layer_sizes`, `activation`, `learning_rate_init`

Each model undergoes feature selection using either:
- **Recursive Feature Elimination (RFE)** for linear models
- **SelectFromModel** for tree-based models


---

## 📁 Project Structure

```
├── README.md             # Project documentation
├── house_data.csv        # Input dataset
├── requirements.txt      # Project libraries
├── train_model.py        # Main training script
├── app.py                # Flask script to setup backend for UI
├── models/               # Saved models and metadata 
├── scalers/              # Saved scalers per model 
├── features/             # Selected features and feature order 
├── react-ui/             # Interactive UI for user
```

---

### 🧹 Dropped Features Before Training

Certain features were dropped prior to model training due to the following reasons:

| Feature         | Reason for Dropping                                      |
|-----------------|----------------------------------------------------------|
| `id`            | Unique identifier; no predictive value                   |
| `date`          | Timestamp of sale; not useful without temporal modeling  |
| `sqft_basement` | Highly correlated with `sqft_living` and `sqft_above`    |
| `yr_renovated`  | Mostly zeros; sparse and not informative                 |

These features either introduced noise, redundancy, or lacked meaningful variance. Dropping them improved model interpretability and reduced overfitting risk.

---

## 📊 Dataset

The dataset contains housing features and sale prices. Key input features include:

- `bedrooms`, `bathrooms`, `floors`, `waterfront`, `view`, `condition`, `grade`
- `yr_built`, `lat`, `long`, `sqft_living`, `sqft_lot`, `zipcode`, etc.
- `price` (target variable)

---

## 🧠 Models Trained

| Model            | R² Score | RMSE       | MAE        | Best Parameters |
|------------------|----------|------------|------------|------------------|
| Linear Regression | 0.634    | 222,106    | 134,890    | —                |
| Ridge Regression  | 0.780    | 172,272    | 90,299     | `max_depth=10`, `min_samples_split=20` |
| Random Forest     | **0.884**| **125,005**| **71,291** | `n_estimators=100`, `max_depth=None`, `min_samples_split=5` |
| SVM               | 0.505    | 258,194    | 135,032    | `C=10`, `kernel=linear`, `gamma=scale` |
| XGBoost           | 0.867    | 133,653    | 71,808     | `learning_rate=0.05`, `max_depth=6`, `n_estimators=200`, `subsample=1.0` |
| Neural Network    | 0.882    | 125,841    | 75,333     | `activation=relu`, `hidden_layer_sizes=(64,64)`, `learning_rate_init=0.005` |

---

## 🏆 Best Performing Model

**Random Forest Regressor**

- **R² Score**: 0.8841
- **RMSE**: 125,004.83
- **MAE**: 71,291.23
- **Best Parameters**:
  - `n_estimators`: 100
  - `max_depth`: None
  - `min_samples_split`: 5
- **Selected Features**:
  - `sqft_living`
  - `waterfront`
  - `grade`
  - `sqft_above`
  - `yr_built`
  - `zipcode`
  - `lat`
  - `long`
  - `sqft_living15`

---

## 🛠️ Setup Instructions
Must follow all steps in order. 3 terminal windows will be used here.

### 1. Install Dependencies
Open a terminal and run
```
pip install -r requirements.txt
```

### 2. Run Training Script
In same terminal run
```
python train_model.py
```
This will:
- Load and clean the dataset
- Visualize distributions and correlations
- Train and evaluate all models
- Save models, scalers, and selected features in respective directories

### 3. Run Backend Script
Open new terminal and run
```
python app.py
```
This will:
- Setup backend
- Load trained models and selected features for each model

### 4. Run UI Script
Must open new terminal and run
```
npm install  # install packages required for UI
```
```
cd react-ui
```
```
npm run dev  # boot up UI locally
```
This will:
- Boot up local UI on http://localhost:5173/. Open this URL on a browser
- Click on 'About' in top bar to see, how to use the UI and get a prediction
---

📂 Output Files

| Folder      | Contents          | 
| ----------- | ----------------- |
| models/ | Trained models (.pkl) and metadata | 
| scalers/ | Scalers used for each model | 
| features/ | Selected features and feature order per model | 
| best_model_summary.json | Summary of top-performing model | 


⚠️ Notes
- Missing values are imputed using column mean.
- Feature selection is model-specific.
- Ensure consistent feature ordering during prediction using saved metadata.
- If browser console shows 500 error, it most likely means python file in Step 3 is not running in parallel to UI.
- if browser console shows 404 error for model selection, it most likely means python file in Step 2 was not executed. Follow steps in order.
