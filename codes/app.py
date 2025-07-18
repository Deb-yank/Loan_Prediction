import streamlit as st
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# ------------------------ DATASET & MODEL ------------------------
df = pd.read_csv('loan.csv')
df = df.drop(df.index[7])  # Remove problematic row

# Handle missing values
mode_imputer = SimpleImputer(strategy='most_frequent')
mean_imputer = SimpleImputer(strategy='mean')
median_imputer = SimpleImputer(strategy='median')

df[['Dependents', 'Self_Employed', 'Married', 'Gender']] = mode_imputer.fit_transform(
    df[['Credit_History', 'Self_Employed', 'Married', 'Gender']]
)
df[['LoanAmount']] = mean_imputer.fit_transform(df[['LoanAmount']])
df[['Loan_Amount_Term', 'Credit_History']] = median_imputer.fit_transform(
    df[['Loan_Amount_Term', 'Credit_History']]
)

# Encode categories
df['Gender'] = df['Gender'].map({'Male': 1, 'Female': 0})
df['Married'] = df['Married'].map({'Yes': 1, 'No': 0})
df['Education'] = df['Education'].map({'Graduate': 1, 'Not Graduate': 0})
df['Self_Employed'] = df['Self_Employed'].map({'Yes': 1, 'No': 0})
df['Loan_Status'] = df['Loan_Status'].map({'Y': 1, 'N': 0})
df['Dependents'] = df['Dependents'].replace('3+', 3).astype(int)

# Feature columns
numerical_cols = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term']
categorical_cols = ['Gender', 'Married', 'Dependents', 'Education']
X = df[numerical_cols + categorical_cols]
y = df['Loan_Status']

# Train/Test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ML Pipeline
pipeline = Pipeline([
    ('preprocessor', ColumnTransformer([
        ('scale', StandardScaler(), numerical_cols),
        ('pass', 'passthrough', categorical_cols)
    ])),
    ('model', LogisticRegression(random_state=0))
])
pipeline.fit(X_train, y_train)

# Accuracy
train_acc = accuracy_score(y_train, pipeline.predict(X_train))
test_acc = accuracy_score(y_test, pipeline.predict(X_test))

# ------------------------ STREAMLIT APP ------------------------
st.set_page_config("Loan Prediction", layout="centered")
st.title("🏦 Smart Loan Approval Predictor")
st.caption("This tool predicts whether a loan will likely be approved based on applicant details.")

with st.expander("🔍 Model Summary", expanded=False):
    st.write("**Class Distribution:**")
    st.write(df['Loan_Status'].value_counts())
    st.write(f"**Training Accuracy:** {train_acc:.2%}")
    st.write(f"**Test Accuracy:** {test_acc:.2%}")

st.subheader("📋 Applicant Information")

# User Inputs
gender = st.radio("Gender", ["Male", "Female"])
married = st.radio("Married", ["Yes", "No"])
dependents = st.selectbox("Number of Dependents", [0, 1, 2, 3])
education = st.radio("Education Level", ["Graduate", "Not Graduate"])

applicant_income = st.number_input("Applicant Income", min_value=0.0, step=100.0, format="%.2f")
coapplicant_income = st.number_input("Coapplicant Income", min_value=0.0, step=100.0, format="%.2f")
loan_amount = st.number_input("Loan Amount (in thousands, recommended: 100–600)", min_value=0.0, step=10.0, format="%.2f")
loan_term = st.number_input("Loan Term (in days, typical: 360)", min_value=30.0, max_value=480.0, step=30.0, format="%.2f")

# 🟡 Warnings if values are unrealistic
if loan_amount < 50 or loan_amount > 700:
    st.warning("⚠️ The loan amount seems unusually low or high. Typical range: 100–600.")
if loan_term < 180:
    st.warning("⚠️ Very short loan terms (e.g., 30 days) may reduce approval chances.")

# Encode user input
gender = 1 if gender == 'Male' else 0
married = 1 if married == 'Yes' else 0
education = 1 if education == 'Graduate' else 0

# Predict
if st.button("🧮 Predict Loan Approval"):
    input_df = pd.DataFrame([[
        applicant_income,
        coapplicant_income,
        loan_amount,
        loan_term,
        gender,
        married,
        dependents,
        education
    ]], columns=numerical_cols + categorical_cols)

    prediction = pipeline.predict(input_df)[0]
    confidence = pipeline.predict_proba(input_df)[0][1]  # Probability of approval

    st.markdown("### 📊 Prediction Result")
    if 0.45 <= confidence <= 0.55:
        st.warning(f"⚠️ Model is unsure. Approval likelihood: {confidence * 100:.2f}%")
    elif prediction == 1:
        st.success(f"✅ Loan Approved with {confidence * 100:.2f}% confidence.")
    else:
        st.error(f"❌ Loan Rejected with {(1 - confidence) * 100:.2f}% confidence.")
