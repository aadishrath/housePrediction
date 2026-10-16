# AdiReality

AdiReality is a React and Flask application that estimates a home's observed price from property details entered in the UI. It uses the project's national Realtor listing dataset as its sole training source. It does not create a month-by-month market trend.

## Price target and status

The dataset's `price` column is status-dependent:

- `sold`: completed sale price.
- `for_sale`: asking/list price.
- `ready_to_build`: asking/list price for a build-ready property.

The form includes **Price context** and the selected status is a model feature. All valid rows from all three statuses are used for model fitting, with status allowing the model to learn which kind of observed price is being estimated. Asking price is not a confirmed transaction price. Since this source snapshot has no reliable current observation date, the result is not an estimate of today's market value.

## Data and feature choices

Place the Kaggle source file at `house_data_new.csv` in the repository root. The 179 MB dataset is intentionally local and ignored by Git; the app does not fetch millions of records from an API on every request.

Raw columns: `brokered_by`, `status`, `price`, `bed`, `bath`, `acre_lot`, `street`, `city`, `state`, `zip_code`, `house_size`, and `prev_sold_date`.

Model inputs:

| Input | Reason |
| --- | --- |
| `status` | Defines completed-sale versus asking-price target semantics. |
| `bed`, `bath` | Basic property layout. |
| `house_size`, `acre_lot` | Interior and land size. |
| `state`, `city`, `zip_code` | Geographic price context. |

Excluded columns:

| Column | Reason |
| --- | --- |
| `price` | Prediction target. |
| `brokered_by` | Broker/agent identifier; not a portable property attribute. |
| `street` | High-cardinality address/record identifier and not needed in the user form. |
| `prev_sold_date` | Not the current listing/sale observation date, incomplete, and contains invalid values. |

Training retains known status, price, and geography records with prices from $10,000 through $100,000,000. It excludes invalid geography and implausible numeric measurements (`bed`/`bath` over 30, home size over 50,000 sq ft, or lot size over 1,000 acres). Missing room/area values are retained and imputed within model training; this is especially useful for `ready_to_build` records. ZIP codes are treated as strings. City, state, ZIP, and status use cross-fitted target encoding, avoiding a huge one-hot matrix and leakage from a row's own target into its training encoding.

## Models and evaluation

The training script compares Linear Regression (baseline), Linear SVR, a PyTorch neural network, histogram gradient boosting, random forest, and XGBoost on the same random 80/20 holdout, stratified by status. Linear SVR uses an epsilon-insensitive SGD solver, which scales to millions of rows. After comparison, it refits every saved prediction model on **all retained rows**, so the holdout is used for evaluation only and no records are reserved from the deployed models. All models learn log-transformed prices; linear, SVR, and neural models use scaled inputs. The neural network and XGBoost use CUDA when available and fall back to CPU; saved artifacts run on CPU for portable inference. The default model is the one with the lowest holdout MAE.

Reported metrics include R², MAE, median absolute error, RMSE, MAPE, mean error, and an 80% absolute-error radius. Since the source has no reliable observation timestamp, the holdout is not time-based. It tests generalization to a random subset of this dataset snapshot, not future market conditions or geographic regions absent from training. Overall scores combine the three status groups, which can have different error profiles.

## Project structure

- `house_data_new.csv` — local training dataset (ignored by Git)
- `train_models.py`, `model_components.py` — quality filtering, preprocessing, six model candidates, evaluation, and the serializable neural-network estimator
- `app.py` — Flask API, input validation, and inference
- `features/input_schema.json` — generated UI controls and validation rules
- `features/selected_features.json` — model input columns
- `features/modeling_notes.json` — included/excluded feature explanations
- `models/` — trained model artifacts, metrics, and training metadata
- `react-ui/` — React interface and model comparison

## Setup and run

Use Python 3.10–3.12 and Node.js 20 or newer. Install dependencies:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
cd react-ui
npm install
cd ..
```

Retrain the models after placing/updating the local dataset:

```powershell
python train_models.py
```

Start the API and UI in separate terminals:

```powershell
python app.py
```

```powershell
cd react-ui
npm run dev
```

Open Vite's local URL (usually `http://localhost:5173`). The development proxy forwards `/api` to Flask at `http://127.0.0.1:5000`.

## Limitations

- The target is the dataset's recorded sale or asking price, selected by status; asking prices need not equal eventual sale prices.
- There is no reliable current listing/sale observation date, so the model cannot establish a current valuation or price trend.
- A random holdout score does not guarantee accuracy for a particular home, location, or price context.
- Dataset records may be duplicated, stale, inaccurate, or unevenly distributed by market. The estimate is educational and is not an appraisal or offer.
