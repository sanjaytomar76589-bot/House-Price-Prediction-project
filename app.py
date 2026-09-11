"""
app.py
------
Streamlit web application for the House Price Prediction project.

Loads the trained ML pipeline (model/house_price_model.pkl) and lets
the user enter house details through simple widgets to get an
estimated price prediction.

Run:
    streamlit run app.py
"""

import os
import joblib
import pandas as pd
import streamlit as st

# ---------------------------------------------------------------
# Paths (relative to this file, so it works on any machine/OS)
# ---------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(SCRIPT_DIR, "model", "house_price_model.pkl")

# ---------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------
st.set_page_config(
    page_title="House Price Prediction",
    page_icon="🏠",
    layout="centered",
)


@st.cache_resource
def load_model():
    """
    Load the trained pipeline from disk.
    Cached with st.cache_resource so the model is loaded only once,
    not on every user interaction.
    """
    if not os.path.exists(MODEL_PATH):
        return None
    data = joblib.load(MODEL_PATH)
    return data


model_data = load_model()

# ---------------------------------------------------------------
# Header
# ---------------------------------------------------------------
st.title("🏠 House Price Prediction")
st.write(
    "A beginner-friendly Machine Learning project that estimates house "
    "prices based on details like area, bedrooms, bathrooms, location, "
    "parking, and house age."
)

if model_data is None:
    st.error(
        "⚠️ Trained model not found.\n\n"
        "Please run the following commands in order before starting the app:\n\n"
        "1. `python generate_dataset.py`\n"
        "2. `python train_model.py`\n\n"
        "Then restart this Streamlit app."
    )
    st.stop()

pipeline = model_data["pipeline"]
model_name = model_data["model_name"]
metrics = model_data["metrics"]
available_locations = model_data["locations"]

st.divider()

# ---------------------------------------------------------------
# Input section
# ---------------------------------------------------------------
st.header("📋 Enter House Details")

col1, col2 = st.columns(2)

with col1:
    area_sqft = st.number_input(
        "Area (sq ft)",
        min_value=200,
        max_value=10000,
        value=1500,
        step=50,
        help="Total built-up area of the house in square feet.",
    )

    bedrooms = st.slider(
        "Bedrooms",
        min_value=1,
        max_value=6,
        value=3,
        help="Number of bedrooms in the house.",
    )

    bathrooms = st.slider(
        "Bathrooms",
        min_value=1,
        max_value=5,
        value=2,
        help="Number of bathrooms in the house.",
    )

with col2:
    location = st.selectbox(
        "Location",
        options=available_locations,
        help="General area/neighborhood type where the house is located.",
    )

    parking = st.selectbox(
        "Parking Available",
        options=["Yes", "No"],
        help="Whether the house has a dedicated parking space.",
    )

    house_age = st.slider(
        "House Age (years)",
        min_value=0,
        max_value=50,
        value=5,
        help="How old the house is, in years.",
    )

st.divider()

# ---------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------
if st.button("🔮 Predict Price", type="primary", use_container_width=True):
    # Build a single-row DataFrame matching the training feature format
    input_df = pd.DataFrame(
        [
            {
                "area_sqft": area_sqft,
                "bedrooms": bedrooms,
                "bathrooms": bathrooms,
                "parking": 1 if parking == "Yes" else 0,
                "house_age": house_age,
                "location": location,
            }
        ]
    )

    try:
        predicted_price = pipeline.predict(input_df)[0]
        predicted_price = max(predicted_price, 0)  # prices can't be negative

        st.success("Estimated House Price")
        st.markdown(
            f"<h2 style='text-align: center; color: #2E7D32;'>"
            f"₹ {predicted_price:,.2f} Lakhs</h2>",
            unsafe_allow_html=True,
        )
    except Exception as e:
        st.error(f"Something went wrong while predicting: {e}")

st.divider()

# ---------------------------------------------------------------
# About the model / project
# ---------------------------------------------------------------
with st.expander("ℹ️ About this Model"):
    st.write(
        f"""
        - **Model used:** {model_name}
        - **Test Set MAE:** {metrics['mae']:.2f} Lakhs
        - **Test Set RMSE:** {metrics['rmse']:.2f} Lakhs
        - **Test Set R² Score:** {metrics['r2']:.4f}

        The model was trained on a **synthetic (artificially generated)**
        dataset created specifically for this educational project. It
        learns realistic patterns such as: larger area, more bedrooms/
        bathrooms, available parking, and newer houses generally increase
        price, while location affects the overall price level.
        """
    )

with st.expander("🛠️ Technologies Used"):
    st.write(
        """
        - **Python** — core programming language
        - **Pandas / NumPy** — data handling and numerical operations
        - **Scikit-learn** — preprocessing, model training & evaluation
        - **Joblib** — saving/loading the trained model pipeline
        - **Streamlit** — interactive web application
        - **Matplotlib / Seaborn** — data visualization (see notebook)
        """
    )

st.caption(
    "⚠️ Disclaimer: This is an educational project. Predictions are "
    "estimates based on a synthetic dataset and should NOT be used for "
    "real financial or property decisions."
)
