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

#Checking for statistical summary of columns
display(Patients.describe())

# COMMAND ----------

#Checking for duplicate rows
Patients.duplicated().sum()

# COMMAND ----------

#Checking for missing or null values in all columns
Patients.isnull().sum()


# COMMAND ----------

#Checking for data types
Patients.dtypes

# COMMAND ----------

# MAGIC %md
# MAGIC _Feature Engineering_

# COMMAND ----------

# MAGIC %md
# MAGIC *For these other extracted columns,  I wanted to see the groups um dealing with, the kind of diagnoses with thier different levels.Also do Different visualisations*

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

# Visualize the distribution of patients across age groups
age_counts = Patients['AgeGroup'].value_counts().sort_index()

plt.figure(figsize=(8, 5))
plt.bar(age_counts.index.astype(str), age_counts.values, color='steelblue', edgecolor='black')

plt.title('Distribution of Patients by Age Group', fontsize=14)
plt.xlabel('Age Group')
plt.ylabel('Number of Patients')

# Show the count on top of each bar
for i, count in enumerate(age_counts.values):
    plt.text(i, count, str(count), ha='center', va='bottom')

plt.tight_layout()
plt.show()

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

# Group the patients by Age Group and Stay Group, then count how many are in each combination
stay_by_age = pd.crosstab(Patients['AgeGroup'], Patients['StayGroup'])
display(stay_by_age)

# Plot the counts as a grouped bar chart (one bar per stay group within each age group)
stay_by_age.plot(kind='bar', figsize=(12, 6))

plt.title('Number of Patients by Age Group and Stay Group')
plt.xlabel('Age Group')
plt.ylabel('Number of Patients')
plt.xticks(rotation=0)
plt.legend(title='Stay Group', bbox_to_anchor=(1.02, 1), loc='upper left')
plt.tight_layout()
plt.show()

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

# Doughnut chart showing the distribution of Blood Pressure Groups within each Age Group

# Build a crosstab of Age Group vs Blood Pressure Group
bp_by_age = pd.crosstab(Patients['AgeGroup'], Patients['BloodPressureGroup'])
display(bp_by_age)

# Plot one doughnut chart per age group
age_groups = bp_by_age.index.tolist()
fig, axes = plt.subplots(1, len(age_groups), figsize=(5 * len(age_groups), 5))

# Ensure axes is iterable even with a single age group
if len(age_groups) == 1:
    axes = [axes]

for ax, age in zip(axes, age_groups):
    counts = bp_by_age.loc[age]
    ax.pie(
        counts,
        labels=counts.index,
        autopct='%1.1f%%',
        startangle=90,
        wedgeprops={'width': 0.4},  # this creates the doughnut hole
        colors=plt.cm.Set3.colors
    )
    ax.set_title(f'{age} Age Group')

plt.suptitle('Blood Pressure Group Distribution by Age Group', fontsize=14)
plt.tight_layout()
plt.show()


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

# Show which number refers to which gender
# le.classes_ lists the categories in alphabetical order, and each gets a number based on that order
# For example, if classes_ is ['Female', 'Male'], then Female = 0 and Male = 1
for number, gender in enumerate(le.classes_):
    print(f'{gender} = {number}')

# Preview the original and encoded columns side by side
display(Patients[['Gender', 'Gender_encoded']].head(10))


# COMMAND ----------


# Target variable: Readmission is binary (Yes/No) -> label encoding maps it to 0/1
Patients['Readmission_encoded'] = le.fit_transform(Patients['Readmission'])

# Show which number belongs to which readmission value
# le.classes_ lists the categories in alphabetical order, so 'No' = 0 and 'Yes' = 1
for number, readmission in enumerate(le.classes_):
    print(f'{readmission} = {number}')

# Admission Type: more than 2 categories with no inherent order -> one-hot encoding
# This avoids implying a false ranking (e.g. 'Emergency' > 'Elective')
# Each admission type gets its own column: 1 if the patient has that type, 0 otherwise
Patients = pd.get_dummies(Patients, columns=['Admission Type'], prefix='AdmissionType', dtype=int)

# Preview the encoded columns
encoded_cols = ['Readmission', 'Readmission_encoded'] + [c for c in Patients.columns if 'AdmissionType' in c]
print('\nAdmission Type columns:', [c for c in Patients.columns if 'AdmissionType' in c])
display(Patients[encoded_cols].head())

# COMMAND ----------

# MAGIC %md
# MAGIC *Train Test Split: in thsi section im going to separate my x features and the y variable,being the target value*
# MAGIC

# COMMAND ----------

# Defining the target column
y = Patients['Readmission']

# Defining the input features
X = Patients.drop(columns=[
    'Readmission',
    'Patient ID',
    'AgeGroup',
    'StayGroup',
    'BloodPressureGroup',
    'BloodSugarGroup'
])

# Display the input features
display(X.head())

# Display the target
display(y.head())

# COMMAND ----------

# MAGIC %md
# MAGIC *Now, i have to train 80% of the data,test 20% and stratify the dataset to ensure that both training and testing sets maintain approximately the same proportion of readmitted and non-readmitted patients as the original dataset.*

# COMMAND ----------

# Import train_test_split
from sklearn.model_selection import train_test_split

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# Check the sizes
print("Training data:", X_train.shape)
print("Testing data:", X_test.shape)

# COMMAND ----------

# MAGIC %md
# MAGIC ##Scale numerical features
# MAGIC I have separated and split the data,I'm going to can scale the numerical columns with
# MAGIC StandardScaler.
# MAGIC StandardScaler changes numerical values so that they have:
# MAGIC - A mean close to 0.
# MAGIC - A standard deviation close to 1.
# MAGIC This helps algorithms such as Logistic Regression, KNN and SVM work effectively when features have different scales.

# COMMAND ----------

# Import StandardScaler
from sklearn.preprocessing import StandardScaler

# List the numerical columns to scale
num_cols = ['Age', 'Blood Pressure', 'Blood Sugar Levels', 'Length of Stay']

# Create the scaler
scaler = StandardScaler()

# Fit the scaler on the training data, then transform it
X_train[num_cols] = scaler.fit_transform(X_train[num_cols])

# Transform the test data using the same scaler (fit only on training data)
X_test[num_cols] = scaler.transform(X_test[num_cols])

# Display the scaled training data
display(X_train.head())

# COMMAND ----------

#checking that the readmission proportions are similar in both sets
print("Original target distribution:")
print(y.value_counts(normalize=True))

print("\nTraining target distribution:")
print(y_train.value_counts(normalize=True))

print("\nTesting target distribution:")
print(y_test.value_counts(normalize=True))