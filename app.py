import streamlit as st
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

st.set_page_config(page_title="Cybersecurity Intrusion Detection", page_icon="🛡️", layout="wide")

NUM_COLS = [
    "network_packet_size",
    "login_attempts",
    "session_duration",
    "ip_reputation_score",
    "failed_logins",
    "unusual_time_access",
]
CAT_COLS = ["protocol_type", "encryption_used", "browser_type"]
FEATURES = NUM_COLS + CAT_COLS
TARGET = "attack_detected"
DATA_FILE = "cybersecurity_intrusion_data.csv"


@st.cache_resource
def train_model():
    data = pd.read_csv(DATA_FILE)
    X = data[FEATURES]
    y = data[TARGET].astype(int)

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])
    preprocess = ColumnTransformer([
        ("num", numeric_pipeline, NUM_COLS),
        ("cat", categorical_pipeline, CAT_COLS),
    ])

    model = Pipeline([
        ("preprocess", preprocess),
        ("classifier", RandomForestClassifier(n_estimators=100, random_state=42)),
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    model.fit(X_train, y_train)
    test_predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, test_predictions)

    # Refit on all available rows for predictions in the app.
    model.fit(X, y)
    return model, data, accuracy, y_test, test_predictions


st.title("🛡️ Cybersecurity Intrusion Detection")
st.write(
    "Enter network-session details to estimate whether the session resembles "
    "normal activity or a detected attack, using a Random Forest model."
)
st.caption("Educational demonstration only — not a production security tool.")

try:
    model, data, test_accuracy, y_test, test_predictions = train_model()
except Exception as exc:
    st.error(f"Could not load or train the model: {exc}")
    st.info("Make sure the CSV file is in the same folder as app.py.")
    st.stop()

m1, m2, m3 = st.columns(3)
m1.metric("Dataset records", f"{len(data):,}")
m2.metric("Features used", len(FEATURES))
m3.metric("Held-out test accuracy", f"{test_accuracy * 100:.2f}%")

st.divider()
st.subheader("Test a network session")

with st.form("prediction_form"):
    left, middle, right = st.columns(3)

    with left:
        packet_size = st.number_input("Network packet size", min_value=0, value=500, step=1)
        protocol = st.selectbox("Protocol type", ["TCP", "UDP", "ICMP"])
        login_attempts = st.number_input("Login attempts", min_value=0, value=3, step=1)
        session_duration = st.number_input("Session duration", min_value=0.0, value=300.0)

    with middle:
        encryption = st.selectbox("Encryption used", ["AES", "DES", "Unknown"])
        ip_score = st.slider("IP reputation score", min_value=0.0, max_value=1.0, value=0.5, step=0.01)
        failed_logins = st.number_input("Failed logins", min_value=0, value=1, step=1)
        browser = st.selectbox("Browser type", ["Chrome", "Edge", "Firefox", "Safari", "Unknown"])

    with right:
        unusual_access = st.selectbox("Unusual time access?", ["No", "Yes"])
        st.write("")
        st.write("")
        submitted = st.form_submit_button("Predict", type="primary", use_container_width=True)

    if submitted:
        sample = pd.DataFrame([{
            "network_packet_size": packet_size,
            "login_attempts": login_attempts,
            "session_duration": session_duration,
            "ip_reputation_score": ip_score,
            "failed_logins": failed_logins,
            "unusual_time_access": 1 if unusual_access == "Yes" else 0,
            "protocol_type": protocol,
            "encryption_used": encryption,
            "browser_type": browser,
        }])
        prediction = int(model.predict(sample)[0])
        probabilities = model.predict_proba(sample)[0]
        confidence = float(np.max(probabilities)) * 100

        st.subheader("Prediction result")
        if prediction == 1:
            st.error("⚠️ The model predicts: Attack detected")
        else:
            st.success("✅ The model predicts: Normal activity")
        st.write(f"Model confidence: **{confidence:.1f}%**")
        st.caption("Confidence is the model's predicted class probability, not a guarantee.")

with st.expander("About this project"):
    st.markdown("""
    **Algorithms compared in the notebook:** Logistic Regression, KNN,
    Decision Tree, Random Forest, SVM and AdaBoost.

    The app uses a preprocessing pipeline with median imputation and scaling
    for numerical fields, most-frequent imputation and one-hot encoding for
    categorical fields, followed by a Random Forest classifier.
    """)
