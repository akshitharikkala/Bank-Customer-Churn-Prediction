
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import os

# =========================================================
# 1. PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Bank Customer Churn Prediction",
    page_icon="🏦",
    layout="wide"
)

st.title("🏦 Bank Customer Churn Prediction Dashboard")
st.write(
    "Predict customer churn, analyze customer behavior, "
    "identify important features and support retention decisions."
)

# =========================================================
# 2. FILE CONFIGURATION
# =========================================================

DATA_PATH = "European_Bank.csv"
MODEL_PATH = "best_model.pkl"

if not os.path.exists(DATA_PATH):
    st.error("European_Bank.csv not found in the project folder.")
    st.stop()

if not os.path.exists(MODEL_PATH):
    st.error("best_model.pkl not found in the project folder.")
    st.stop()

# =========================================================
# 3. LOAD DATASET AND MACHINE LEARNING MODEL
# =========================================================

@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

try:
    df = load_data()
    churn_model = load_model()
    st.success("Machine Learning Model Loaded Successfully!")

except Exception as e:
    st.error(f"Loading error: {e}")
    st.stop()

# Clean column names
df.columns = df.columns.str.strip()

required_columns = [
    "Exited", "Geography", "Gender", "Age",
    "CreditScore", "Tenure", "Balance",
    "NumOfProducts", "HasCrCard",
    "IsActiveMember", "EstimatedSalary"
]

missing_dataset_columns = [
    col for col in required_columns
    if col not in df.columns
]

if missing_dataset_columns:
    st.error(
        f"Dataset is missing these columns: "
        f"{missing_dataset_columns}"
    )
    st.stop()

# =========================================================
# 4. IDENTIFY MODEL INPUT FEATURES
# =========================================================

def get_model_features(model, data):

    features = getattr(model, "feature_names_in_", None)

    if features is None and hasattr(model, "named_steps"):

        for step in model.named_steps.values():

            if hasattr(step, "feature_names_in_"):
                features = step.feature_names_in_
                break

    if features is not None:
        return list(features)

    # Fallback for models without saved feature names
    return [
        col for col in data.columns
        if col not in [
            "CustomerId", "Surname", "Exited", "Age_Group"
        ]
    ]


model_features = get_model_features(churn_model, df)

# Detect whether the saved model expects raw or one-hot encoded data
encoded_df = pd.get_dummies(
    df,
    columns=["Geography", "Gender"],
    drop_first=True
)

raw_mode = all(
    feature in df.columns
    for feature in model_features
)

encoded_mode = all(
    feature in encoded_df.columns
    for feature in model_features
)

if not raw_mode and not encoded_mode:
    st.error(
        "The model's expected input columns do not match "
        "the dataset. Check the original model training code."
    )
    st.write("Model features:", model_features)
    st.write("Dataset features:", list(df.columns))
    st.stop()

# =========================================================
# 5. PREPARE CUSTOMER INPUT FOR PREDICTION
# =========================================================

def prepare_model_input(customer_data):

    input_df = pd.DataFrame([customer_data])

    if raw_mode:

        prepared = input_df.reindex(
            columns=model_features
        )

    else:

        categorical_columns = [
            col for col in ["Geography", "Gender"]
            if col in input_df.columns
        ]

        prepared = pd.get_dummies(
            input_df,
            columns=categorical_columns,
            drop_first=True
        )

        prepared = prepared.reindex(
            columns=model_features,
            fill_value=0
        )

    return prepared


def predict_churn(customer_data):

    prepared = prepare_model_input(customer_data)

    prediction = int(churn_model.predict(prepared)[0])

    if not hasattr(churn_model, "predict_proba"):
        raise ValueError(
            "The saved model does not support predict_proba."
        )

    probabilities = churn_model.predict_proba(prepared)[0]

    classes = list(churn_model.classes_)

    if 1 not in classes:
        raise ValueError(
            "The model does not contain churn class 1."
        )

    churn_index = classes.index(1)

    probability = float(probabilities[churn_index])

    return prediction, probability


def risk_category(probability):

    if probability >= 0.60:
        return "HIGH RISK"

    elif probability >= 0.30:
        return "MEDIUM RISK"

    return "LOW RISK"


# =========================================================
# 6. DASHBOARD OVERVIEW
# =========================================================

st.divider()
st.header("1. Dashboard Overview")

total_customers = len(df)
churned = int((df["Exited"] == 1).sum())
retained = int((df["Exited"] == 0).sum())

churn_rate = (
    churned / total_customers * 100
    if total_customers else 0
)

col1, col2, col3, col4 = st.columns(4)

col1.metric("Total Customers", f"{total_customers:,}")
col2.metric("Churned Customers", f"{churned:,}")
col3.metric("Retained Customers", f"{retained:,}")
col4.metric("Churn Rate", f"{churn_rate:.2f}%")

# =========================================================
# 7. CUSTOMER CHURN DISTRIBUTION
# =========================================================

st.subheader("2. Customer Churn Distribution")

fig, ax = plt.subplots(figsize=(8, 4))

ax.bar(
    ["Retained", "Churned"],
    [retained, churned]
)

ax.set_ylabel("Number of Customers")
ax.set_title("Customer Churn Analysis")

st.pyplot(fig)
plt.close(fig)

# =========================================================
# 8. CUSTOMER DATASET
# =========================================================

st.divider()
st.header("3. Customer Dataset")

st.dataframe(
    df.head(20),
    width="stretch"
)

st.write("Total records:", total_customers)

# =========================================================
# 9. ADVANCED CUSTOMER ANALYTICS
# =========================================================

st.divider()
st.header("4. Advanced Customer Analytics")

# Geography analysis
st.subheader("4.1 Churn Rate by Geography")

geo_churn = (
    df.groupby("Geography")["Exited"].mean() * 100
)

st.bar_chart(geo_churn)

# Gender analysis
st.subheader("4.2 Churn Rate by Gender")

gender_churn = (
    df.groupby("Gender")["Exited"].mean() * 100
)

st.bar_chart(gender_churn)

# Age group analysis
st.subheader("4.3 Churn Rate by Age Group")

age_df = df.copy()

age_df["Age_Group"] = pd.cut(
    age_df["Age"],
    bins=[0, 25, 35, 45, 55, 100],
    labels=["18-25", "26-35", "36-45", "46-55", "55+"]
)

age_churn = (
    age_df.groupby(
        "Age_Group",
        observed=False
    )["Exited"].mean() * 100
)

st.bar_chart(age_churn)

# Active customer analysis
st.subheader("4.4 Churn by Customer Activity")

activity_churn = (
    df.groupby("IsActiveMember")["Exited"].mean() * 100
)

activity_churn.index = [
    "Inactive" if value == 0 else "Active"
    for value in activity_churn.index
]

st.bar_chart(activity_churn)

# Number of products analysis
st.subheader("4.5 Churn by Number of Products")

product_churn = (
    df.groupby("NumOfProducts")["Exited"].mean() * 100
)

st.bar_chart(product_churn)

st.success("Advanced Customer Analytics Completed!")

# =========================================================
# 10. INDIVIDUAL CUSTOMER CHURN PREDICTION
# =========================================================

st.divider()
st.header("5. Customer Churn Risk Prediction")

st.write(
    "Enter customer information to calculate "
    "the probability of customer churn."
)

# Only show original customer features required by the model
if raw_mode:
    form_features = model_features.copy()
else:
    form_features = [
        feature for feature in model_features
        if feature in df.columns
    ]

    if any(
        feature.startswith("Geography_")
        for feature in model_features
    ):
        form_features.append("Geography")

    if any(
        feature.startswith("Gender_")
        for feature in model_features
    ):
        form_features.append("Gender")

form_features = list(dict.fromkeys(form_features))

form_features = [
    feature for feature in form_features
    if feature in df.columns
    and feature not in [
        "CustomerId", "Surname", "Exited", "Age_Group"
    ]
]

with st.form("customer_prediction_form"):

    st.subheader("Enter Customer Details")

    customer_input = {}

    col1, col2 = st.columns(2)

    for i, feature in enumerate(form_features):

        current_col = col1 if i % 2 == 0 else col2

        with current_col:

            if feature in ["Geography", "Gender"]:

                options = sorted(
                    df[feature].dropna().unique().tolist()
                )

                customer_input[feature] = st.selectbox(
                    feature,
                    options
                )

            elif feature in [
                "HasCrCard", "IsActiveMember"
            ]:

                customer_input[feature] = st.selectbox(
                    feature,
                    [0, 1],
                    format_func=lambda x:
                    "Yes" if x == 1 else "No"
                )

            elif pd.api.types.is_numeric_dtype(df[feature]):

                minimum = float(df[feature].min())
                maximum = float(df[feature].max())
                default = float(df[feature].median())

                if pd.api.types.is_integer_dtype(df[feature]):

                    customer_input[feature] = st.number_input(
                        feature,
                        min_value=int(minimum),
                        max_value=int(maximum),
                        value=int(default),
                        step=1
                    )

                else:

                    customer_input[feature] = st.number_input(
                        feature,
                        min_value=minimum,
                        max_value=maximum,
                        value=default
                    )

    submit = st.form_submit_button(
        "Predict Churn Risk",
        width="stretch"
    )

if submit:

    try:

        prediction, probability = predict_churn(
            customer_input
        )

        st.divider()
        st.subheader("Prediction Results")

        result1, result2 = st.columns(2)

        result1.metric(
            "Churn Probability",
            f"{probability * 100:.2f}%"
        )

        result2.metric(
            "Predicted Status",
            "CHURN" if prediction == 1 else "RETAINED"
        )

        st.progress(probability)

        risk = risk_category(probability)

        if risk == "HIGH RISK":

            st.error(
                "HIGH RISK: Priority customer retention required."
            )

        elif risk == "MEDIUM RISK":

            st.warning(
                "MEDIUM RISK: Customer needs attention."
            )

        else:

            st.success(
                "LOW RISK: Customer is less likely to leave."
            )

        st.info(
            "Recommendation: Review customer engagement "
            "and consider suitable retention offers."
        )

    except Exception as e:

        st.error(f"Prediction error: {e}")

# =========================================================
# 11. FEATURE IMPORTANCE ANALYSIS
# =========================================================

st.divider()
st.header("6. Feature Importance Analysis")

def get_feature_importance(model):

    estimator = model

    if hasattr(model, "named_steps"):

        estimator = list(model.named_steps.values())[-1]

    if not hasattr(estimator, "feature_importances_"):
        return None, None

    importance = estimator.feature_importances_

    feature_names = None

    if hasattr(model, "get_feature_names_out"):

        try:
            feature_names = model.get_feature_names_out()
        except Exception:
            pass

    if feature_names is None:
        feature_names = model_features

    if len(feature_names) != len(importance):
        feature_names = [
            f"Feature {i + 1}"
            for i in range(len(importance))
        ]

    return feature_names, importance


try:

    feature_names, importance = get_feature_importance(
        churn_model
    )

    if feature_names is not None:

        importance_df = pd.DataFrame({
            "Feature": feature_names,
            "Importance": importance
        })

        importance_df = importance_df.sort_values(
            "Importance",
            ascending=False
        ).head(10)

        fig, ax = plt.subplots(figsize=(10, 5))

        ax.barh(
            importance_df["Feature"],
            importance_df["Importance"]
        )

        ax.set_title("Top 10 Important Features")
        ax.set_xlabel("Importance")
        ax.invert_yaxis()

        st.pyplot(fig)
        plt.close(fig)

        st.dataframe(
            importance_df,
            width="stretch"
        )

        st.success("Feature Importance Analysis Completed!")

    else:

        st.info(
            "Feature importance is unavailable for this model."
        )

except Exception as e:

    st.warning(f"Feature importance unavailable: {e}")

# =========================================================
# 12. WHAT-IF SCENARIO SIMULATOR
# =========================================================

st.divider()
st.header("7. What-if Scenario Simulator")

st.write(
    "Change customer details to understand how different "
    "situations affect churn risk."
)

# Create a baseline customer using dataset median/mode
baseline_customer = {}

for feature in form_features:

    if pd.api.types.is_numeric_dtype(df[feature]):

        baseline_customer[feature] = float(
            df[feature].median()
        )

    else:

        baseline_customer[feature] = df[feature].mode()[0]

with st.form("scenario_form"):

    st.subheader("Change Customer Scenario")

    scenario_customer = baseline_customer.copy()

    s_col1, s_col2 = st.columns(2)

    if "Age" in scenario_customer:

        with s_col1:

            scenario_customer["Age"] = st.slider(
                "Customer Age",
                int(df["Age"].min()),
                int(df["Age"].max()),
                int(baseline_customer["Age"])
            )

    if "IsActiveMember" in scenario_customer:

        with s_col2:

            scenario_customer["IsActiveMember"] = st.selectbox(
                "Customer Activity",
                [0, 1],
                index=int(baseline_customer["IsActiveMember"]),
                format_func=lambda x:
                "Active" if x == 1 else "Inactive"
            )

    if "NumOfProducts" in scenario_customer:

        with s_col1:

            scenario_customer["NumOfProducts"] = st.selectbox(
                "Number of Products",
                sorted(df["NumOfProducts"].unique().tolist()),
                index=sorted(
                    df["NumOfProducts"].unique().tolist()
                ).index(
                    int(baseline_customer["NumOfProducts"])
                )
            )

    if "Geography" in scenario_customer:

        with s_col2:

            scenario_customer["Geography"] = st.selectbox(
                "Geography",
                sorted(df["Geography"].unique().tolist()),
                index=sorted(
                    df["Geography"].unique().tolist()
                ).index(
                    baseline_customer["Geography"]
                )
            )

    simulate = st.form_submit_button(
        "Run What-if Simulation",
        width="stretch"
    )

if simulate:

    try:

        baseline_prediction, baseline_probability = (
            predict_churn(baseline_customer)
        )

        scenario_prediction, scenario_probability = (
            predict_churn(scenario_customer)
        )

        st.subheader("Scenario Comparison")

        compare1, compare2 = st.columns(2)

        compare1.metric(
            "Baseline Churn Risk",
            f"{baseline_probability * 100:.2f}%"
        )

        compare2.metric(
            "Scenario Churn Risk",
            f"{scenario_probability * 100:.2f}%",
            delta=f"{(scenario_probability - baseline_probability) * 100:.2f}%"
        )

        st.write(
            "Baseline Status:",
            "CHURN" if baseline_prediction == 1 else "RETAINED"
        )

        st.write(
            "Scenario Status:",
            "CHURN" if scenario_prediction == 1 else "RETAINED"
        )

        st.write(
            "Scenario Risk:",
            risk_category(scenario_probability)
        )

        if scenario_probability < baseline_probability:

            st.success(
                "The changed customer details reduced "
                "the predicted churn probability."
            )

        elif scenario_probability > baseline_probability:

            st.warning(
                "The changed customer details increased "
                "the predicted churn probability."
            )

        else:

            st.info(
                "The scenario did not change the predicted "
                "churn probability."
            )

    except Exception as e:

        st.error(f"Scenario prediction error: {e}")

# =========================================================
# 13. CUSTOMER CHURN REPORT DOWNLOAD
# =========================================================

st.divider()
st.header("8. Customer Churn Prediction Report")

if st.button("Generate Customer Churn Report"):

    try:

        report_df = df.copy()

        report_inputs = report_df[
            form_features
        ].copy()

        predictions = churn_model.predict(
            prepare_model_input(
                report_inputs.iloc[0].to_dict()
            )
        ) if len(report_inputs) == 1 else None

        # Prepare all customer records using the same encoding
        if raw_mode:

            model_report_input = report_inputs.reindex(
                columns=model_features
            )

        else:

            model_report_input = pd.get_dummies(
                report_inputs,
                columns=[
                    col for col in ["Geography", "Gender"]
                    if col in report_inputs.columns
                ],
                drop_first=True
            ).reindex(
                columns=model_features,
                fill_value=0
            )

        report_df["Predicted_Churn"] = (
            churn_model.predict(model_report_input)
        )

        report_probabilities = (
            churn_model.predict_proba(model_report_input)
        )

        churn_index = list(
            churn_model.classes_
        ).index(1)

        report_df["Churn_Probability"] = (
            report_probabilities[:, churn_index] * 100
        ).round(2)

        report_df["Risk_Category"] = (
            report_df["Churn_Probability"].apply(
                lambda x: risk_category(x / 100)
            )
        )

        st.dataframe(
            report_df.head(20),
            width="stretch"
        )

        csv_data = report_df.to_csv(
            index=False
        ).encode("utf-8")

        st.download_button(
            label="Download Churn Prediction Report",
            data=csv_data,
            file_name="Bank_Customer_Churn_Report.csv",
            mime="text/csv"
        )

        st.success("Customer Churn Report Generated Successfully!")

    except Exception as e:

        st.error(f"Report generation error: {e}")

# =========================================================
# 14. FOOTER
# =========================================================

st.divider()

st.caption(
    "Bank Customer Churn Prediction System | "
    "Python | Machine Learning | Streamlit | Pandas"
)

st.caption(
    "Developed as a Data Analytics and Machine Learning Project."
)
