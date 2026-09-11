# 🏠 House Price Prediction Using Machine Learning

A beginner-friendly, end-to-end Machine Learning project that predicts
estimated house prices based on features like area, bedrooms, bathrooms,
location, parking, and house age — with an interactive Streamlit web app.

---

## 📌 Project Overview

This project demonstrates a complete ML workflow: generating a dataset,
training a regression model, evaluating it, and deploying it as a live
web application. A user enters house details in the web app and instantly
gets an estimated price prediction from the trained model.

It is designed as a **student/portfolio project** — simple enough to
understand fully, but structured the way a real ML project would be.

---

## ✨ Features

- Clean, interactive Streamlit UI with number inputs, sliders, and a dropdown
- Real ML pipeline: preprocessing + model trained and evaluated on real (synthetic) data
- Compares **Linear Regression** vs **Random Forest Regressor** and automatically keeps the better one
- Evaluation using MAE, RMSE, and R² — no hardcoded/fake accuracy numbers
- Model saved with Joblib and loaded directly by the web app
- Ready to push to GitHub and deploy on Streamlit Community Cloud
- Includes a Jupyter notebook with EDA (exploratory data analysis) and visualizations

---

## 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| **Python** | Core programming language |
| **Pandas** | Loading and manipulating tabular data |
| **NumPy** | Numerical operations |
| **Scikit-learn** | Preprocessing, model training, evaluation |
| **Matplotlib / Seaborn** | Data visualization (used in the notebook) |
| **Joblib** | Saving and loading the trained model pipeline |
| **Streamlit** | Building and deploying the interactive web app |

---

## 🤖 Machine Learning Approach

The workflow follows a standard supervised regression pipeline:

```
Dataset → Preprocessing → Train/Test Split → Model Training → Evaluation → Prediction
```

1. **Dataset**: A synthetic dataset is generated with realistic relationships between features and price, plus random noise (see below).
2. **Preprocessing**: Numeric features (`area_sqft`, `bedrooms`, `bathrooms`, `parking`, `house_age`) are scaled with `StandardScaler`. The categorical feature (`location`) is one-hot encoded with `OneHotEncoder`. Both are combined using a `ColumnTransformer`.
3. **Train/Test Split**: 80% of the data is used for training, 20% is held out for testing — the split happens **before** any preprocessing is fit, so there is no data leakage.
4. **Model Training**: Two models are trained — `LinearRegression` (simple baseline) and `RandomForestRegressor` (captures non-linear patterns).
5. **Evaluation**: Both models are scored on the untouched test set using **MAE**, **RMSE**, and **R²**. The better-performing model (by R²) is automatically selected.
6. **Prediction**: The winning pipeline (preprocessing + model, bundled together) is saved with Joblib and loaded directly by the Streamlit app for live predictions.

> ⚠️ The exact MAE/RMSE/R² numbers printed by `train_model.py` and shown in the app are calculated from the actual trained model on your machine — they are **not** hardcoded, and may vary slightly between runs since the dataset generation and model training involve randomness.

---

## 📊 Dataset

**This project uses a synthetic (artificially generated) dataset created for educational purposes.**

A real, freely-licensed, deployment-reliable house price dataset with
exactly these features (area, bedrooms, bathrooms, location, parking,
house age) is hard to guarantee across environments. To keep the project
100% reliable to run and deploy anywhere, `generate_dataset.py` creates
a 3,000-row dataset where:

- Price increases with area, bedrooms, bathrooms, and parking availability
- Price decreases as house age increases (depreciation)
- Each location has a different price multiplier (e.g., "City Center" costs more than "Rural")
- Random noise is added so the relationship isn't perfectly clean (just like real-world data)

| Column | Description |
|---|---|
| `area_sqft` | Total built-up area in square feet |
| `bedrooms` | Number of bedrooms |
| `bathrooms` | Number of bathrooms |
| `location` | Neighborhood type (Downtown, Suburb, Rural, City Center, Industrial Area) |
| `parking` | 1 if parking available, 0 otherwise |
| `house_age` | Age of the house in years |
| `price` | Target variable — price in **Lakhs (INR)** |

---

## 📁 Project Structure

```
house-price-prediction/
│
├── app.py                     # Streamlit web application
├── train_model.py             # Trains, evaluates, and saves the ML model
├── generate_dataset.py        # Generates the synthetic dataset
├── requirements.txt           # Python dependencies
├── README.md                  # Project documentation (this file)
├── .gitignore                 # Files/folders excluded from git
│
├── data/
│   └── house_prices.csv       # Generated dataset
│
├── model/
│   └── house_price_model.pkl  # Saved trained pipeline (preprocessing + model)
│
└── notebooks/
    └── analysis.ipynb         # EDA, visualizations, and model exploration
```

---

## 💻 Installation (Run Locally)

### 1. Clone the repository

```bash
git clone <repository-url>
cd house-price-prediction
```

### 2. Create and activate a virtual environment

```bash
python -m venv venv
```

**Windows:**
```bash
venv\Scripts\activate
```

**macOS/Linux:**
```bash
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Generate the dataset

```bash
python generate_dataset.py
```

This creates `data/house_prices.csv`.

### 5. Train the model

```bash
python train_model.py
```

This prints the evaluation metrics (MAE, RMSE, R²) and saves the trained
pipeline to `model/house_price_model.pkl`.

### 6. Run the Streamlit app

```bash
streamlit run app.py
```

The app will open automatically in your browser at `http://localhost:8501`.

> 💡 The repository already includes a pre-generated `data/house_prices.csv`
> and a pre-trained `model/house_price_model.pkl`, so you can also skip
> straight to step 6 and run the app immediately. Steps 4–5 are only
> needed if you want to regenerate the dataset or retrain the model
> yourself.

---

## ☁️ Deployment on Streamlit Community Cloud

1. **Create a GitHub repository** and push this entire project folder to it (make sure `data/house_prices.csv` and `model/house_price_model.pkl` are included in the commit — see the note in `.gitignore`).
2. Go to [Streamlit Community Cloud](https://streamlit.io/cloud) and sign in (you can sign in with your GitHub account).
3. Click **"New app"**.
4. **Connect your GitHub account** if you haven't already, and grant access to your repository.
5. **Select the repository** you just pushed (e.g., `your-username/house-price-prediction`).
6. Set the branch (usually `main`) and set the main file path to **`app.py`**.
7. Click **"Deploy"**.

Streamlit Cloud will install everything listed in `requirements.txt` and
launch `app.py`. After a minute or two, you'll get a live public URL for
your app.

### Important deployment notes

- Streamlit Cloud only runs `app.py` — it does **not** automatically run `generate_dataset.py` or `train_model.py`. This is why the dataset and trained model files are committed to the repository rather than gitignored.
- If you change `generate_dataset.py` or `train_model.py` and want the deployed app to reflect a newly trained model, you must re-run both scripts **locally**, then commit and push the updated `data/house_prices.csv` and `model/house_price_model.pkl` files.
- Keep `requirements.txt` versions reasonably flexible (as done here with `>=`) to avoid dependency resolution failures on the cloud.

---

## 🧪 Example

**Sample input:**

| Field | Value |
|---|---|
| Area | 1800 sq ft |
| Bedrooms | 3 |
| Bathrooms | 2 |
| Location | Suburb |
| Parking | Yes |
| House Age | 5 years |

**Output:** an estimated price such as *"₹ XX.XX Lakhs"* — the exact
number depends entirely on the model that gets trained on your machine,
since training involves randomness (data split, Random Forest
initialization, etc.).

---

## 🚀 Future Improvements

- Replace the synthetic dataset with a larger real-world housing dataset
- Add more diverse and realistic locations
- Add more features (e.g., number of floors, furnishing status, proximity to amenities)
- Try additional models (Gradient Boosting, XGBoost) and compare performance
- Add hyperparameter tuning (GridSearchCV / RandomizedSearchCV)
- Improve the UI with charts showing how each input affects the predicted price

---

## ⚠️ Disclaimer

This is an **educational project** built for learning and portfolio
purposes. The dataset is synthetic, and predictions are rough estimates
only — they should **not** be used for real financial, investment, or
property valuation decisions.
