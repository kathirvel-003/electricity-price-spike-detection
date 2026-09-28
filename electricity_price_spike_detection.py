import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Electricity Market Price Spike Detection",
    page_icon="⚡",
    layout="wide"
)

# ============================================================
# TITLE
# ============================================================

st.title("⚡ Electricity Market Price Spike Detection")

st.write(
    "A machine learning application for detecting electricity "
    "market price spikes using demand, renewable generation, "
    "temperature, wind speed and time-related factors."
)

# ============================================================
# LOAD DATASET
# ============================================================

@st.cache_data
def load_data():

    file_name = "electricity_market_price_spike_dataset.csv"

    data = pd.read_csv(file_name)

    return data


# ============================================================
# TRAIN MACHINE LEARNING MODEL
# ============================================================

@st.cache_resource
def train_model(data):

    features = [
        "Hour",
        "Demand_MW",
        "Renewable_Generation_MW",
        "Temperature_C",
        "Wind_Speed_kmh",
        "Day_of_Week",
        "Is_Weekend"
    ]

    X = data[features]

    y = data["Price_Spike"]

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    # Random Forest model
    model = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        class_weight="balanced"
    )

    # Train model
    model.fit(X_train, y_train)

    # Prediction
    y_pred = model.predict(X_test)

    # Evaluation
    accuracy = accuracy_score(y_test, y_pred)

    precision = precision_score(
        y_test,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        y_pred,
        zero_division=0
    )

    cm = confusion_matrix(
        y_test,
        y_pred
    )

    return (
        model,
        features,
        accuracy,
        precision,
        recall,
        f1,
        cm
    )


# ============================================================
# LOAD DATA WITH ERROR HANDLING
# ============================================================

try:

    df = load_data()

except FileNotFoundError:

    st.error(
        "❌ Dataset file not found."
    )

    st.warning(
        "Make sure this file is uploaded to the same GitHub repository:"
    )

    st.code(
        "electricity_market_price_spike_dataset.csv"
    )

    st.stop()

except Exception as e:

    st.error(
        "❌ Error loading dataset."
    )

    st.code(
        str(e)
    )

    st.stop()


# ============================================================
# CHECK REQUIRED COLUMNS
# ============================================================

required_columns = [
    "Date",
    "Hour",
    "Demand_MW",
    "Renewable_Generation_MW",
    "Temperature_C",
    "Wind_Speed_kmh",
    "Electricity_Price",
    "Price_Spike",
    "Day_of_Week",
    "Is_Weekend"
]

missing_columns = [
    column
    for column in required_columns
    if column not in df.columns
]

if missing_columns:

    st.error(
        "❌ Required columns are missing from the dataset."
    )

    st.write(
        "Missing columns:"
    )

    st.write(
        missing_columns
    )

    st.stop()


# ============================================================
# TRAIN MODEL
# ============================================================

(
    model,
    features,
    accuracy,
    precision,
    recall,
    f1,
    cm
) = train_model(df)


# ============================================================
# SIDEBAR NAVIGATION
# ============================================================

st.sidebar.title("⚡ Navigation")

page = st.sidebar.radio(
    "Select Page",
    [
        "Dashboard",
        "Dataset",
        "Prediction",
        "Model Performance"
    ]
)


# ============================================================
# DASHBOARD
# ============================================================

if page == "Dashboard":

    st.header("📊 Electricity Market Dashboard")

    # Calculate values
    total_records = len(df)

    price_spikes = int(
        df["Price_Spike"].sum()
    )

    normal_prices = (
        total_records - price_spikes
    )

    average_price = (
        df["Electricity_Price"].mean()
    )

    maximum_price = (
        df["Electricity_Price"].max()
    )

    # Metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.metric(
            "Total Records",
            f"{total_records:,}"
        )

    with col2:

        st.metric(
            "Price Spikes",
            f"{price_spikes:,}"
        )

    with col3:

        st.metric(
            "Normal Prices",
            f"{normal_prices:,}"
        )

    with col4:

        st.metric(
            "Average Price",
            f"{average_price:.2f}"
        )

    st.divider()

    # Price trend
    st.subheader(
        "📈 Electricity Market Price Trend"
    )

    fig, ax = plt.subplots(
        figsize=(12, 5)
    )

    ax.plot(
        df["Electricity_Price"],
        linewidth=1
    )

    ax.set_title(
        "Electricity Market Price"
    )

    ax.set_xlabel(
        "Observation"
    )

    ax.set_ylabel(
        "Electricity Price"
    )

    ax.grid(
        alpha=0.25
    )

    st.pyplot(
        fig,
        clear_figure=True
    )

    plt.close(fig)

    # Price spike distribution
    st.subheader(
        "🚨 Price Spike Distribution"
    )

    spike_counts = (
        df["Price_Spike"]
        .value_counts()
        .reindex(
            [0, 1],
            fill_value=0
        )
    )

    fig2, ax2 = plt.subplots(
        figsize=(8, 4)
    )

    ax2.bar(
        [
            "Normal Price",
            "Price Spike"
        ],
        spike_counts.values
    )

    ax2.set_title(
        "Normal Price vs Price Spike"
    )

    ax2.set_ylabel(
        "Number of Records"
    )

    st.pyplot(
        fig2,
        clear_figure=True
    )

    plt.close(fig)

    # Additional information
    st.subheader(
        "📌 Project Information"
    )

    st.write(
        """
        This project uses a Random Forest machine learning model
        to identify possible electricity market price spikes.

        Important factors include:

        • Electricity demand

        • Renewable generation

        • Temperature

        • Wind speed

        • Hour of the day

        • Day of the week

        • Weekend status
        """
    )


# ============================================================
# DATASET PAGE
# ============================================================

elif page == "Dataset":

    st.header(
        "📁 Electricity Market Dataset"
    )

    st.write(
        f"Dataset contains **{df.shape[0]:,} rows** "
        f"and **{df.shape[1]} columns**."
    )

    st.subheader(
        "Dataset Preview"
    )

    st.dataframe(
        df,
        use_container_width=True
    )

    st.subheader(
        "📊 Statistical Summary"
    )

    st.dataframe(
        df.describe(),
        use_container_width=True
    )

    st.subheader(
        "🔎 Dataset Information"
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "Rows",
            f"{df.shape[0]:,}"
        )

    with col2:

        st.metric(
            "Columns",
            df.shape[1]
        )

    with col3:

        st.metric(
            "Missing Values",
            int(df.isnull().sum().sum())
        )


# ============================================================
# PREDICTION PAGE
# ============================================================

elif page == "Prediction":

    st.header(
        "⚡ Electricity Price Spike Prediction"
    )

    st.write(
        "Enter the current market conditions below."
    )

    col1, col2 = st.columns(2)

    # Left column
    with col1:

        hour = st.slider(
            "Hour of Day",
            min_value=0,
            max_value=23,
            value=18
        )

        demand = st.number_input(
            "Electricity Demand (MW)",
            min_value=0.0,
            value=4000.0,
            step=50.0
        )

        renewable = st.number_input(
            "Renewable Generation (MW)",
            min_value=0.0,
            value=300.0,
            step=10.0
        )

        temperature = st.number_input(
            "Temperature (°C)",
            min_value=-20.0,
            max_value=60.0,
            value=30.0,
            step=1.0
        )

    # Right column
    with col2:

        wind = st.number_input(
            "Wind Speed (km/h)",
            min_value=0.0,
            value=10.0,
            step=1.0
        )

        day = st.selectbox(
            "Day of Week",
            options=[
                0,
                1,
                2,
                3,
                4,
                5,
                6
            ],
            index=2,
            format_func=lambda x: [
                "Monday",
                "Tuesday",
                "Wednesday",
                "Thursday",
                "Friday",
                "Saturday",
                "Sunday"
            ][x]
        )

        weekend = st.selectbox(
            "Is Weekend?",
            options=[
                0,
                1
            ],
            format_func=lambda x:
                "Yes" if x == 1 else "No"
        )

    st.divider()

    # Prediction button
    if st.button(
        "🔍 Detect Price Spike",
        type="primary"
    ):

        input_data = pd.DataFrame({

            "Hour": [hour],

            "Demand_MW": [demand],

            "Renewable_Generation_MW": [
                renewable
            ],

            "Temperature_C": [
                temperature
            ],

            "Wind_Speed_kmh": [
                wind
            ],

            "Day_of_Week": [
                day
            ],

            "Is_Weekend": [
                weekend
            ]
        })

        # Prediction
        prediction = int(
            model.predict(
                input_data
            )[0]
        )

        # Probability
        probability = float(
            model.predict_proba(
                input_data
            )[0][1]
        )

        st.subheader(
            "Prediction Result"
        )

        if prediction == 1:

            st.error(
                f"🚨 PRICE SPIKE DETECTED"
            )

            st.metric(
                "Spike Probability",
                f"{probability:.2%}"
            )

            st.warning(
                "The model predicts that the "
                "current conditions may result "
                "in an electricity price spike."
            )

        else:

            st.success(
                "✅ NORMAL PRICE"
            )

            st.metric(
                "Spike Probability",
                f"{probability:.2%}"
            )

            st.info(
                "The model predicts normal "
                "electricity market price conditions."
            )


# ============================================================
# MODEL PERFORMANCE
# ============================================================

elif page == "Model Performance":

    st.header(
        "🤖 Machine Learning Model Performance"
    )

    st.write(
        "Random Forest Classifier"
    )

    # Metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.metric(
            "Accuracy",
            f"{accuracy:.2%}"
        )

    with col2:

        st.metric(
            "Precision",
            f"{precision:.2%}"
        )

    with col3:

        st.metric(
            "Recall",
            f"{recall:.2%}"
        )

    with col4:

        st.metric(
            "F1 Score",
            f"{f1:.2%}"
        )

    st.divider()

    # Confusion Matrix
    st.subheader(
        "📊 Confusion Matrix"
    )

    fig3, ax3 = plt.subplots(
        figsize=(6, 5)
    )

    ax3.imshow(
        cm
    )

    ax3.set_title(
        "Random Forest Confusion Matrix"
    )

    ax3.set_xlabel(
        "Predicted"
    )

    ax3.set_ylabel(
        "Actual"
    )

    ax3.set_xticks(
        [0, 1]
    )

    ax3.set_yticks(
        [0, 1]
    )

    ax3.set_xticklabels(
        [
            "Normal",
            "Spike"
        ]
    )

    ax3.set_yticklabels(
        [
            "Normal",
            "Spike"
        ]
    )

    for i in range(
        cm.shape[0]
    ):

        for j in range(
            cm.shape[1]
        ):

            ax3.text(
                j,
                i,
                cm[i, j],
                ha="center",
                va="center"
            )

    st.pyplot(
        fig3,
        clear_figure=True
    )

    plt.close(fig3)

    # Feature importance
    st.subheader(
        "📌 Feature Importance"
    )

    importance = pd.Series(
        model.feature_importances_,
        index=features
    ).sort_values(
        ascending=True
    )

    fig4, ax4 = plt.subplots(
        figsize=(9, 5)
    )

    importance.plot(
        kind="barh",
        ax=ax4
    )

    ax4.set_title(
        "Random Forest Feature Importance"
    )

    ax4.set_xlabel(
        "Importance"
    )

    st.pyplot(
        fig4,
        clear_figure=True
    )

    plt.close(fig4)


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Electricity Market Price Spike Detection | "
    "Machine Learning + Streamlit"
)
