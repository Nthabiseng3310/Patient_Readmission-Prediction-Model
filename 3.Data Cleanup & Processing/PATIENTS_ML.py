# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "6"
# ///
# MAGIC %md
# MAGIC **_Task 1,Data processing and cleanUp_**

# COMMAND ----------

#Importing Liabraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

# COMMAND ----------

#calling the Patients table
Patients=spark.table('patients_ml2.readmission.pr')
#converting the table to pandas
Patients=Patients.toPandas()
display(Patients)

# COMMAND ----------

# MAGIC %md
# MAGIC **Data inspection**

# COMMAND ----------

#Checking for the number of Rows and Columns
Patients.shape

# COMMAND ----------

#Checking for duplicate rows
duplicate_rows = Patients.duplicated().sum()
print(f"Number of duplicate rows: {duplicate_rows}")

#Checking for duplicates based on patient identifier (if any)
#Display duplicate rows if any exist
if duplicate_rows > 0:
    display(Patients[Patients.duplicated(keep=False)].sort_values(by=Patients.columns[0]))

# COMMAND ----------

#Checking for missing or null values in all columns
missing_values = Patients.isnull().sum()
print("Missing or null values per column:")
print(missing_values)

#Total missing values across the entire dataset
total_missing = Patients.isnull().sum().sum()
print(f"\nTotal missing or null values in the dataset: {total_missing}")

#Display columns with missing values
print("\nColumns with missing values:")
cols_with_missing = missing_values[missing_values > 0]
if len(cols_with_missing) > 0:
    display(cols_with_missing)
else:
    print("No missing or null values found in any column.")

# COMMAND ----------

#Checking for data types
Patients.dtypes

# COMMAND ----------

# MAGIC %md
# MAGIC _Feature Engineering_

# COMMAND ----------

# Create age group buckets using pd.cut
# Adjust the column name if your age column is named differently
age_column = 'Age'  # change this to match your dataset's age column name

# Define bin edges and labels for the age groups
bins = [0, 12, 18, 34, 50, 95]
labels = ['Kids', 'Youth', 'Adults', 'Elders', 'Pensioners']

Patients['AgeGroup'] = pd.cut(
    Patients[age_column],
    bins=bins,
    labels=labels,
    right=True,          # intervals are (left, right]
    include_lowest=True   # first interval includes the lowest value (0)
)

# Show the distribution of patients across age groups
print('Age group distribution:')
print(Patients['AgeGroup'].value_counts().sort_index())

# Preview the new column
display(Patients[[age_column, 'AgeGroup']].head(10))

# COMMAND ----------

# Group length of stay into categories from short to long
# Adjust the column name if your stay column is named differently
stay_column = 'Length of Stay'  # change this to match your dataset's stay column name

# Define bin edges (in days) and labels for the stay groups
stay_bins = [0, 3, 7, 14, 30, 1000]
stay_labels = ['Very Short (0-3 days)', 'Short (4-7 days)', 'Medium (8-14 days)', 'Long (15-30 days)', 'Very Long (30+ days)']

Patients['StayGroup'] = pd.cut(
    Patients[stay_column],
    bins=stay_bins,
    labels=stay_labels,
    right=True,           # intervals are (left, right]
    include_lowest=True    # first interval includes the lowest value (0)
)

# Show the distribution of patients across stay groups
print('Length of stay group distribution:')
print(Patients['StayGroup'].value_counts().sort_index())

# Preview the new column
display(Patients[[stay_column, 'StayGroup']].head(10))

# COMMAND ----------

# Group blood pressure into Low, Normal, and High categories
# Adjust the column name if your blood pressure column is named differently
bp_column = 'Blood Pressure'  # change this to match your dataset's blood pressure column name

# Define bin edges and labels for blood pressure groups
# Bins are based on systolic blood pressure (mmHg):
#   Low:      below 90
#   Normal:   90-120
#   Pre-high: 121-139  (elevated / stage 1 hypertension)
#   High:     140 and above
bp_bins = [0, 89, 120, 139, 300]
bp_labels = ['Low (<90)', 'Normal (90-120)', 'Pre-high (121-139)', 'High (140+)']

Patients['BloodPressureGroup'] = pd.cut(
    Patients[bp_column],
    bins=bp_bins,
    labels=bp_labels,
    right=True,
    include_lowest=True
)

# Show the distribution of patients across blood pressure groups
print('Blood pressure group distribution:')
print(Patients['BloodPressureGroup'].value_counts().sort_index())

# Preview the new column
display(Patients[[bp_column, 'BloodPressureGroup']].head(10))

# COMMAND ----------

# Group blood sugar levels into Low, Normal, Pre-diabetic, and High categories
# Adjust the column name if your blood sugar column is named differently
bs_column = 'Blood Sugar Levels'

# Define bin edges (mg/dL) and labels for blood sugar groups
# Based on fasting blood sugar levels:
#   Low:          below 70
#   Normal:       70-99
#   Pre-diabetic: 100-125
#   Diabetic:     126 and above
bs_bins = [0, 69, 99, 125, 1000]
bs_labels = ['Low (<70)', 'Normal (70-99)', 'Pre-diabetic (100-125)', 'Diabetic (126+)']

Patients['BloodSugarGroup'] = pd.cut(
    Patients[bs_column],
    bins=bs_bins,
    labels=bs_labels,
    right=True,
    include_lowest=True
)

# Show the distribution of patients across blood sugar groups
print('Blood sugar group distribution:')
print(Patients['BloodSugarGroup'].value_counts().sort_index())

# Preview the new column
display(Patients[[bs_column, 'BloodSugarGroup']].head(10))

# COMMAND ----------

# MAGIC %md
# MAGIC ## LabelEncoder is used for Gender and the target because they are binary (2 categories,Male and female)
# MAGIC # One-hot encoding is used for Admission Type because it has more than 2 categories
# MAGIC # and no natural ordering exists between them/_Hierachy_#

# COMMAND ----------

# Encode categorical variables into numeric form
le = LabelEncoder()

# Gender: binary (e.g. Male/Female) -> label encoding maps to 0/1
Patients['Gender_encoded'] = le.fit_transform(Patients['Gender'])
display('Gender_encoded')


# COMMAND ----------


# Target variable: binary (e.g. Yes/No) -> label encoding maps to 0/1
Patients['Readmission_encoded'] = le.fit_transform(Patients['Readmission'])

# Admission Type: more than 2 categories with no inherent order -> one-hot encoding
# This avoids implying a false ranking (e.g. 'Emergency' > 'Elective')
Patients = pd.get_dummies(Patients, columns=['Admission Type'], prefix='AdmissionType', dtype=int)

# Preview the encoded columns
encoded_cols = [c for c in Patients.columns if 'encoded' in c or 'AdmissionType' in c]
print("Encoded columns:", encoded_cols)
display(Patients[encoded_cols].head())