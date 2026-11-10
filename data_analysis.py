
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import joblib

# ==========================================
# 1. LOAD DATASET
# ==========================================

df = pd.read_csv("European_Bank.csv")

print("===== BANK CUSTOMER CHURN DATASET =====")

# Dataset Shape
print("\nDataset Shape:")
print(df.shape)

# First 5 Rows
print("\nFirst 5 Rows:")
print(df.head())

# Column Names
print("\nColumn Names:")
print(df.columns.tolist())

# Missing Values
print("\nMissing Values:")
print(df.isnull().sum())

# Churn Distribution
print("\nChurn Distribution:")
print(df["Exited"].value_counts())

# Churn Percentage
print("\nChurn Percentage:")
print(df["Exited"].value_counts(normalize=True) * 100)


# ==========================================
# 2. CHURN DISTRIBUTION
# ==========================================

plt.figure(figsize=(6, 4))

sns.countplot(x="Exited", data=df)

plt.title("Customer Churn Distribution")
plt.xlabel("Exited (0 = No, 1 = Yes)")
plt.ylabel("Number of Customers")

plt.show()


# ==========================================
# 3. AGE VS CHURN
# ==========================================

plt.figure(figsize=(7, 5))

sns.boxplot(x="Exited", y="Age", data=df)

plt.title("Age Distribution by Customer Churn")
plt.xlabel("Exited (0 = No, 1 = Yes)")
plt.ylabel("Age")

plt.show()


# ==========================================
# 4. GENDER VS CHURN
# ==========================================

plt.figure(figsize=(6, 4))

sns.countplot(
    x="Gender",
    hue="Exited",
    data=df
)

plt.title("Customer Churn by Gender")
plt.xlabel("Gender")
plt.ylabel("Number of Customers")

plt.legend(
    title="Exited",
    labels=["No", "Yes"]
)

plt.show()


# ==========================================
# 5. GEOGRAPHY VS CHURN
# ==========================================

plt.figure(figsize=(7, 4))

sns.countplot(
    x="Geography",
    hue="Exited",
    data=df
)

plt.title("Customer Churn by Geography")
plt.xlabel("Geography")
plt.ylabel("Number of Customers")

plt.legend(
    title="Exited",
    labels=["No", "Yes"]
)

plt.show()


# ==========================================
# 6. CREDIT SCORE VS CHURN
# ==========================================

plt.figure(figsize=(7, 5))

sns.boxplot(
    x="Exited",
    y="CreditScore",
    data=df
)

plt.title("Credit Score Distribution by Customer Churn")
plt.xlabel("Exited (0 = No, 1 = Yes)")
plt.ylabel("Credit Score")

plt.show()


# ==========================================
# 7. BALANCE VS CHURN
# ==========================================

plt.figure(figsize=(7, 5))

sns.boxplot(
    x="Exited",
    y="Balance",
    data=df
)

plt.title("Balance Distribution by Customer Churn")
plt.xlabel("Exited (0 = No, 1 = Yes)")
plt.ylabel("Account Balance")

plt.show()


# ==========================================
# 8. NUMBER OF PRODUCTS VS CHURN
# ==========================================

plt.figure(figsize=(7, 5))

sns.countplot(
    x="NumOfProducts",
    hue="Exited",
    data=df
)

plt.title("Customer Churn by Number of Products")
plt.xlabel("Number of Products")
plt.ylabel("Number of Customers")

plt.legend(
    title="Exited",
    labels=["No", "Yes"]
)

plt.show()


# ==========================================
# 9. ACTIVE MEMBER VS CHURN
# ==========================================

plt.figure(figsize=(7, 5))

sns.countplot(
    x="IsActiveMember",
    hue="Exited",
    data=df
)

plt.title("Customer Churn by Active Membership")
plt.xlabel("Active Member (0 = No, 1 = Yes)")
plt.ylabel("Number of Customers")

plt.legend(
    title="Exited",
    labels=["No", "Yes"]
)

plt.show()


# ==========================================
# 10. TENURE VS CHURN
# ==========================================

plt.figure(figsize=(7, 5))

sns.countplot(
    x="Tenure",
    hue="Exited",
    data=df
)

plt.title("Customer Churn by Tenure")
plt.xlabel("Tenure (Years)")
plt.ylabel("Number of Customers")

plt.legend(
    title="Exited",
    labels=["No", "Yes"]
)

plt.show()


# ==========================================
# 11. ESTIMATED SALARY VS CHURN
# ==========================================

plt.figure(figsize=(7, 5))

sns.boxplot(
    x="Exited",
    y="EstimatedSalary",
    data=df
)

plt.title("Estimated Salary Distribution by Customer Churn")
plt.xlabel("Exited (0 = No, 1 = Yes)")
plt.ylabel("Estimated Salary")

plt.show()


# ==========================================
# 12. CARD TYPE VS CHURN
# ==========================================

plt.figure(figsize=(7, 4))

sns.countplot(
    x="HasCrCard",
    hue="Exited",
    data=df
)

plt.title("Customer Churn by Card Type")
plt.xlabel("Card Type")
plt.ylabel("Number of Customers")

plt.legend(
    title="Exited",
    labels=["No", "Yes"]
)

plt.show()


# ==========================================
# ANALYSIS COMPLETED
# ==========================================

print("\n==========================================")
print("ALL EDA ANALYSIS COMPLETED SUCCESSFULLY")
print("==========================================")

# ==========================================
# 10. DATA PREPROCESSING
# ==========================================

print("\n===== DATA PREPROCESSING =====")

# Remove unnecessary columns
df_ml = df.drop(columns=["RowNumber", "CustomerId", "Surname"], errors="ignore")

# Convert Gender to numeric
df_ml["Gender"] = df_ml["Gender"].map({
    "Male": 1,
    "Female": 0
})

# Convert categorical Geography using One-Hot Encoding
df_ml = pd.get_dummies(
    df_ml,
    columns=["Geography"],
    drop_first=True
)

# Convert boolean columns to integers
df_ml = df_ml.astype(int)

print("\nProcessed Dataset:")
print(df_ml.head())

print("\nProcessed Columns:")
print(df_ml.columns.tolist())

print("\nMissing Values After Preprocessing:")
print(df_ml.isnull().sum())

print("\n===== PREPROCESSING COMPLETED =====")

# ==========================================
# 11. TRAIN-TEST SPLIT
# ==========================================

from sklearn.model_selection import train_test_split

print("\n===== TRAIN-TEST SPLIT =====")

# Separate features and target
X = df_ml.drop("Exited", axis=1)
y = df_ml["Exited"]

# Split dataset: 80% training, 20% testing
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("Training data shape:", X_train.shape)
print("Testing data shape:", X_test.shape)

print("\n===== TRAIN-TEST SPLIT COMPLETED =====")


# ==========================================
# 10. DATA PREPROCESSING
# ==========================================

print("\n===== DATA PREPROCESSING =====")

# Display columns
print("\nOriginal Columns:")
print(df.columns.tolist())

# Remove unnecessary columns
columns_to_drop = ["RowNumber", "CustomerId", "Surname"]

# Drop only columns that actually exist
df = df.drop(columns=[col for col in columns_to_drop if col in df.columns])

print("\nColumns after removing unnecessary columns:")
print(df.columns.tolist())


# ==========================================
# 11. CONVERT CATEGORICAL VARIABLES
# ==========================================

# Convert Gender and Geography into numerical values
df = pd.get_dummies(
    df,
    columns=["Gender", "Geography"],
    drop_first=True
)

print("\nData after encoding:")
print(df.head())


# ==========================================
# 12. SEPARATE FEATURES AND TARGET
# ==========================================

X = df.drop("Exited", axis=1)
y = df["Exited"]

print("\nFeatures Shape:")
print(X.shape)

print("\nTarget Shape:")
print(y.shape)


# ==========================================
# 13. TRAIN TEST SPLIT
# ==========================================

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining Data Shape:")
print(X_train.shape)

print("\nTesting Data Shape:")
print(X_test.shape)


# ==========================================
# 14. FEATURE SCALING
# ==========================================

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\nFeature Scaling Completed Successfully")


# ==========================================
# 15. LOGISTIC REGRESSION MODEL
# ==========================================

from sklearn.linear_model import LogisticRegression

logistic_model = LogisticRegression(
    max_iter=1000,
    random_state=42
)

logistic_model.fit(X_train_scaled, y_train)

print("\nLogistic Regression Model Trained Successfully")


# ==========================================
# 16. RANDOM FOREST MODEL
# ==========================================

from sklearn.ensemble import RandomForestClassifier

random_forest_model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    class_weight="balanced"
)

random_forest_model.fit(X_train, y_train)

print("Random Forest Model Trained Successfully")


# ==========================================
# 17. MODEL PREDICTIONS
# ==========================================

y_pred_logistic = logistic_model.predict(X_test_scaled)

y_pred_rf = random_forest_model.predict(X_test)

print("\nPredictions Generated Successfully")


# ==========================================
# 18. MODEL EVALUATION
# ==========================================

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)


# Logistic Regression Results
print("\n==========================================")
print("LOGISTIC REGRESSION RESULTS")
print("==========================================")

print("Accuracy:",
      accuracy_score(y_test, y_pred_logistic))

print("Precision:",
      precision_score(y_test, y_pred_logistic))

print("Recall:",
      recall_score(y_test, y_pred_logistic))

print("F1 Score:",
      f1_score(y_test, y_pred_logistic))

print("\nClassification Report:")
print(classification_report(y_test, y_pred_logistic))


# Random Forest Results
print("\n==========================================")
print("RANDOM FOREST RESULTS")
print("==========================================")

print("Accuracy:",
      accuracy_score(y_test, y_pred_rf))

print("Precision:",
      precision_score(y_test, y_pred_rf))

print("Recall:",
      recall_score(y_test, y_pred_rf))

print("F1 Score:",
      f1_score(y_test, y_pred_rf))

print("\nClassification Report:")
print(classification_report(y_test, y_pred_rf))


# ==========================================
# 19. CONFUSION MATRIX
# ==========================================

plt.figure(figsize=(6, 5))

sns.heatmap(
    confusion_matrix(y_test, y_pred_rf),
    annot=True,
    fmt="d",
    cmap="Blues"
)

plt.title("Random Forest - Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.show()


# ==========================================
# 20. FEATURE IMPORTANCE
# ==========================================

feature_importance = pd.Series(
    random_forest_model.feature_importances_,
    index=X.columns
).sort_values(ascending=False)

print("\n===== FEATURE IMPORTANCE =====")
print(feature_importance)


plt.figure(figsize=(10, 6))

feature_importance.head(10).plot(kind="bar")

plt.title("Top 10 Important Features")
plt.xlabel("Features")
plt.ylabel("Importance")

plt.xticks(rotation=45)
plt.tight_layout()

plt.show()


# ==========================================
# 21. FINAL MESSAGE
# ==========================================

print("\n==========================================")
print("MACHINE LEARNING ANALYSIS COMPLETED")
print("==========================================")


# ==========================================
# 22. CUSTOMER CHURN RISK PREDICTION
# ==========================================

print("\n===== CUSTOMER CHURN RISK PREDICTION =====")

# Example customer
customer = X_test.iloc[[0]]

# Predict churn
prediction = random_forest_model.predict(customer)[0]

# Predict churn probability
probability = random_forest_model.predict_proba(customer)[0][1]

print("\nCustomer Churn Prediction:")

if prediction == 1:
    print("Prediction: CUSTOMER LIKELY TO CHURN")
else:
    print("Prediction: CUSTOMER LIKELY TO STAY")

print(f"Churn Probability: {probability:.2%}")


# ==========================================
# 23. RISK CATEGORY
# ==========================================

if probability < 0.30:
    risk = "LOW RISK"
elif probability < 0.60:
    risk = "MEDIUM RISK"
else:
    risk = "HIGH RISK"

print("Risk Category:", risk)

print("\n==========================================")
print("CUSTOMER RISK PREDICTION COMPLETED")
print("==========================================")

# ==========================================
# MACHINE LEARNING PREPROCESSING
# ==========================================

# Features:
# CreditScore, Gender, Age, Tenure, Balance
# NumOfProducts, HasCrCard, IsActiveMember
# EstimatedSalary

# Target:
# Exited

# Training Data Shape:
# (8000, ...)

# Testing Data Shape:
# (2000, ...)

# ==========================================
# MACHINE LEARNING MODEL DEVELOPMENT
# ==========================================

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

print("\n===== MODEL DEVELOPMENT STARTED =====")

# Select features and target
X = df.drop(columns=["Exited", "CustomerId", "Surname"], errors="ignore")
y = df["Exited"]

# Identify column types
numeric_features = X.select_dtypes(include=["int64", "float64"]).columns
categorical_features = X.select_dtypes(include=["object"]).columns

# Preprocessing
numeric_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("num", numeric_transformer, numeric_features),
    ("cat", categorical_transformer, categorical_features)
])

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Define models
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(
        random_state=42, class_weight="balanced"
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        class_weight="balanced"
    )
}

# Train and evaluate
for name, model in models.items():

    pipeline = Pipeline([
        ("preprocessing", preprocessor),
        ("model", model)
    ])

    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)

    print("\nModel:", name)
    print("Accuracy:", accuracy_score(y_test, predictions))
    print(classification_report(y_test, predictions))
    

print("\n===== MODEL DEVELOPMENT COMPLETED SUCCESSFULLY =====")

# ==========================================
# BEST MODEL SELECTION
# ==========================================

best_model = None
best_model_name = ""
best_f1 = 0

for name, model in models.items():

    pipeline = Pipeline([
        ("preprocessing", preprocessor),
        ("model", model)
    ])

    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    f1 = f1_score(y_test, predictions)

    print("\nModel:", name)
    print("Accuracy:", accuracy)
    print("F1 Score:", f1)

    print(classification_report(y_test, predictions))

    if f1 > best_f1:
        best_f1 = f1
        best_model = pipeline
        best_model_name = name

# Save best model
joblib.dump(best_model, "best_model.pkl")

print("\nBEST MODEL:", best_model_name)
print("BEST F1 SCORE:", best_f1)
print("MODEL SAVED SUCCESSFULLY!")


import streamlit as st
import pandas as pd
import joblib
import os

# Page configuration
st.set_page_config(
    page_title="Bank Customer Churn Prediction",
    page_icon="🏦",
    layout="wide"
)

# Load saved model
MODEL_PATH = "best_model.pkl"

st.title("🏦 Bank Customer Churn Prediction System")
st.write("Predict customer churn risk using Machine Learning.")

if not os.path.exists(MODEL_PATH):
    st.error("Model file not found. Please train and save best_model.pkl first.")
    st.stop()

model = joblib.load(MODEL_PATH)

st.success("Machine Learning model loaded successfully!")

st.header("Customer Information")

# Customer inputs
col1, col2 = st.columns(2)

with col1:
    credit_score = st.number_input(
        "Credit Score", 300, 900, 650
    )

    gender = st.selectbox(
        "Gender", ["Male", "Female"]
    )

    age = st.number_input(
        "Age", 18, 100, 35
    )

    tenure = st.number_input(
        "Tenure (Years)", 0, 10, 5
    )

    balance = st.number_input(
        "Account Balance", min_value=0.0, value=50000.0
    )

with col2:
    products = st.number_input(
        "Number of Products", 1, 4, 2
    )

    credit_card = st.selectbox(
        "Has Credit Card?", [1, 0],
        format_func=lambda x: "Yes" if x == 1 else "No"
    )

    active = st.selectbox(
        "Is Active Member?", [1, 0],
        format_func=lambda x: "Yes" if x == 1 else "No"
    )

    salary = st.number_input(
        "Estimated Salary", min_value=0.0, value=50000.0
    )

# Prediction button
if st.button("Predict Customer Churn", type="primary"):

    customer = pd.DataFrame([{
        "CreditScore": credit_score,
        "Gender": gender,
        "Age": age,
        "Tenure": tenure,
        "Balance": balance,
        "NumOfProducts": products,
        "HasCrCard": credit_card,
        "IsActiveMember": active,
        "EstimatedSalary": salary
    }])

    try:
        prediction = model.predict(customer)[0]

        if hasattr(model, "predict_proba"):
            probability = model.predict_proba(customer)[0][1]
        else:
            st.error("The saved model does not support probability prediction.")
            st.stop()

        risk = probability * 100

        st.header("Prediction Results")

        st.metric(
            "Customer Churn Probability",
            f"{risk:.2f}%"
        )

        st.progress(float(probability))

        if risk >= 60:
            st.error("HIGH RISK - Customer may leave the bank.")
        elif risk >= 30:
            st.warning("MEDIUM RISK - Customer needs attention.")
        else:
            st.success("LOW RISK - Customer is likely to stay.")

        st.write(
            "Predicted Status:",
            "Churn" if prediction == 1 else "Retained"
        )

        st.info(
            "Recommendation: Review customer engagement "
            "and consider suitable retention offers."
        )

    except Exception as e:
        st.error(f"Prediction error: {e}")
        st.write("Check that the input columns match the trained model.")

