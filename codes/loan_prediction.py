

import pandas as pd
# Import the pandas library for data manipulation and analysis

import numpy as np
# Import NumPy for numerical operations, especially arrays and mathematical functions

import matplotlib.pyplot as plt
# Import Matplotlib's pyplot for plotting graphs and charts

import seaborn as sns
# Import seaborn for advanced data visualisation (built on top of matplotlib)

from sklearn.model_selection import train_test_split
# Import train_test_split to split the dataset into training and testing sets

from sklearn.linear_model import LogisticRegression
# Import LogisticRegression model for performing logistic regression

from sklearn.metrics import accuracy_score

from sklearn.impute import SimpleImputer
# Import SimpleImputer to fill in missing values (e.g. using mean, median, or most frequent value)

from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
# Import encoders: OneHotEncoder for nominal categorical data, OrdinalEncoder for ordered categorical data



from sklearn.compose import ColumnTransformer

# Import ColumnTransformer to apply different preprocessing steps to specific columns

df= pd.read_csv('loan.csv') # load the dataset

df.head() # dissplay the first 5 rows

df.info  # Get comprehensive information about the dataset

df.describe()  # Display basic statistics for numerical columns

df.isnull().sum()  # Check for missing values in each column

from sklearn.impute import SimpleImputer

df = df.drop(df.index[7])

# Create imputers
mode_imputer = SimpleImputer(strategy='most_frequent')
mean_imputer = SimpleImputer(strategy='mean')
median_imputer = SimpleImputer(strategy='median')

# Apply imputers to categorical columns
# Impute categorical columns with mode (most frequent value)
df[['Dependents', 'Self_Employed', 'Married', 'Gender']] = mode_imputer.fit_transform(df[['Credit_History', 'Self_Employed', 'Married', 'Gender']])

# Apply imputers to numerical columns
df[['LoanAmount']] = mean_imputer.fit_transform(df[['LoanAmount']])
df[['Loan_Amount_Term','Credit_History']] = median_imputer.fit_transform(df[['Loan_Amount_Term','Credit_History']])

df.isnull().sum()





# Create an explicit copy to avoid the warning
df = df.copy()

#
# Binary encoding for categorical variables
# Convert categorical variables to numerical (0 and 1)
df['Gender'] = df['Gender'].map({'Male': 1, 'Female': 0})
df['Married'] = df['Married'].map({'Yes': 1, 'No': 0})
df['Education'] = df['Education'].map({'Graduate': 1, 'Not Graduate': 0})
df['Self_Employed'] = df['Self_Employed'].map({'Yes': 1, 'No': 0})
df['Loan_Status'] = df['Loan_Status'].map({'Y': 1, 'N': 0})

# Handle 'Dependents' column - convert '3+' to 3 and make it integer
df['Dependents'] = df['Dependents'].replace('3+', 3).astype(int)

df.head()

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
# Define columns
numerical_cols = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term']
categorical_cols = ['Gender', 'Married', 'Dependents', 'Education']

X= df.drop('Loan_Status', axis=1) # Features (all columns except target)
y=df['Loan_Status']   # Target variable (what we want to predict)


# Split the data into training and testing sets
X_train, X_test, y_train, y_test= train_test_split(
    X,y,
    test_size=0.2,   # 20% for testing, 80% for training
    random_state=42   # For reproducible results
)



# Create complete ML pipeline
pipeline = Pipeline([
    ('preprocessor', ColumnTransformer([
             # StandardScaler transforms features to have mean=0 and std=1
        ('num_scaler', StandardScaler(), numerical_cols),



        # 'passthrough' means keep these columns as-is
        ('cat_passthrough', 'passthrough', categorical_cols)
    ], remainder='drop')),
      # LogisticRegression for binary classification (Loan approved/rejected)
    # random_state=0 ensures reproducible results
    ('model', LogisticRegression(random_state=0))
])

pipeline.fit(X_train, y_train)
# 1. Calculate scaling parameters on training data
# 2. Transform training data
# 3. Train the logistic regression model

# Make predictions
train_predictions = pipeline.predict(X_train)
test_predictions = pipeline.predict(X_test)

# Evaluate the model
print("Training Accuracy:", accuracy_score(y_train, train_predictions))
print("Test Accuracy:", accuracy_score(y_test, test_predictions))

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Generate the confusion matrix
cm = confusion_matrix(y_test, test_predictions)

# Display it
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap='Blues')
plt.title("Confusion Matrix")
plt.show()

from sklearn.metrics import classification_report, confusion_matrix

# Detailed classification metrics
print("Classification Report:")
print(classification_report(y_test, test_predictions))
# Confusion matrix
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, test_predictions))



