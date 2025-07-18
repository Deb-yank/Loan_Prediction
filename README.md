# Loan Prediction Machine Learning Model

A machine learning project that predicts loan approval status using logistic regression. This model analyzes various applicant features to determine the likelihood of loan approval.

## Overview

This project implements a binary classification model to predict whether a loan application will be approved or rejected. The model uses logistic regression with proper data preprocessing and evaluation metrics.

## Features

- **Data Preprocessing**: Handles missing values using appropriate imputation strategies
- **Feature Engineering**: Converts categorical variables to numerical format
- **Scalable Pipeline**: Uses scikit-learn pipelines for reproducible preprocessing and modeling
- **Model Evaluation**: Comprehensive evaluation with accuracy scores, confusion matrix, and classification report

## Dataset

The model expects a CSV file named `loan.csv` with the following columns:

### Input Features:
- `Gender`: Male/Female
- `Married`: Yes/No
- `Dependents`: Number of dependents (0, 1, 2, 3+)
- `Education`: Graduate/Not Graduate
- `Self_Employed`: Yes/No
- `ApplicantIncome`: Applicant's income
- `CoapplicantIncome`: Co-applicant's income
- `LoanAmount`: Loan amount requested
- `Loan_Amount_Term`: Loan term in months
- `Credit_History`: Credit history (1.0/0.0)

### Target Variable:
- `Loan_Status`: Y (approved) / N (rejected)

## Requirements

```
pandas
numpy
matplotlib
seaborn
scikit-learn
```

## Installation

1. Clone or download the project files
2. Install required packages:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```
3. Ensure your dataset `loan.csv` is in the same directory as the script

## Usage

Run the script to train and evaluate the model:

```bash
python loan_prediction.py
```

The script will:
1. Load and explore the dataset
2. Handle missing values using imputation
3. Preprocess categorical and numerical features
4. Split data into training and testing sets
5. Train a logistic regression model
6. Evaluate performance and display results

## Model Pipeline

The model uses a scikit-learn pipeline with the following steps:

1. **Data Preprocessing**:
   - **Missing Value Imputation**:
     - Categorical variables: Most frequent value (mode)
     - Numerical variables: Mean or median
   - **Feature Encoding**:
     - Binary encoding for categorical variables
     - Numerical scaling using StandardScaler

2. **Model Training**:
   - Logistic Regression with default parameters
   - 80/20 train-test split
   - Random state set for reproducibility

## Performance Metrics

The model provides several evaluation metrics:

- **Accuracy Score**: Overall prediction accuracy
- **Confusion Matrix**: Visual representation of true vs predicted classifications
- **Classification Report**: Detailed precision, recall, and F1-scores
- **Training vs Test Accuracy**: To check for overfitting

## Data Preprocessing Details

### Missing Value Handling:
- **Categorical columns** (`Dependents`, `Self_Employed`, `Married`, `Gender`): Filled with most frequent value
- **Numerical columns**:
  - `LoanAmount`: Filled with mean
  - `Loan_Amount_Term`, `Credit_History`: Filled with median

### Feature Encoding:
- Binary categorical variables mapped to 0/1
- `Dependents` column: '3+' converted to 3
- Numerical features standardized using StandardScaler

## File Structure

```
loan_prediction/
├── loan_prediction.py    # Main script
├── loan.csv             # Dataset (required)
└── README.md           # This file
```

## Results

The model outputs:
- Training and test accuracy scores
- Confusion matrix visualization
- Detailed classification report with precision, recall, and F1-scores

## Future Improvements

Potential enhancements for the model:
- Feature selection and engineering
- Hyperparameter tuning
- Cross-validation
- Trying different algorithms (Random Forest, XGBoost, etc.)
- Handling class imbalance if present
- Feature importance analysis

## Notes

- The script removes row index 7 from the dataset (may need adjustment based on your data)
- Random state is set to 42 for train-test split and 0 for the model to ensure reproducible results
- The model assumes the target variable is binary (approved/rejected)

## License

This project is for educational and demonstration purposes.
