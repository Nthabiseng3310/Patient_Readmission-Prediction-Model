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

# COMMAND ----------

# MAGIC %md
# MAGIC #Task 2: EDA,this is the process of examining and visualising data to understand patterns, relationships, trends and unusual observations before building a machine learning model.

# COMMAND ----------

# MAGIC %md
# MAGIC /In our project, we want to understand:
# MAGIC - Which patients are being readmitted?
# MAGIC - What is the distribution of patient ages?
# MAGIC - Does length of hospital stay relate to readmission?
# MAGIC - Do previous admissions influence readmission?
# MAGIC - Are there relationships between blood pressure, blood sugar and readmission?/

# COMMAND ----------

# Count readmission outcomes

readmission_counts = Patients['Readmission'].value_counts()

display(readmission_counts)

# COMMAND ----------

# DBTITLE 1,Cell 32
# MAGIC %pip install seaborn
# MAGIC #understanding the balance between patients who were readmitted and those who were not,-we have many more No outcomes than Yes outcomes, the dataset is imbalanced.
# MAGIC
# MAGIC # Plot readmission outcomes
# MAGIC
# MAGIC import seaborn as sns
# MAGIC
# MAGIC plt.figure(figsize=(7, 5))
# MAGIC
# MAGIC sns.countplot(data=Patients, x='Readmission')
# MAGIC
# MAGIC plt.title('Distribution of Patient Readmission')
# MAGIC plt.xlabel('Readmission Outcome')
# MAGIC plt.ylabel('Number of Patients')
# MAGIC
# MAGIC plt.show()

# COMMAND ----------

#by dealing with imbalance,i will use the weight class imbalance to tell the machine learning model to give more importance to the minority class (Readmission = Yes)This encourages the model to pay more attention to patients who are readmitted
class_weight='balanced'

# COMMAND ----------

# Calculate the percentage of each class
#The target variable showed class imbalance, with approximately 71.1% of patients not readmitted and 28.9% readmitted. This imbalance may cause the model to favour the majority class. Class weights,will therefore be investigated during model development to improve the identification of readmitted patients.

class_balance = Patients['Readmission'].value_counts(normalize=True) * 100

display(class_balance)

# COMMAND ----------

#Generating descriptive columns

display(Patients.describe())

# COMMAND ----------

# Select numerical columns;We want to understand the characteristics of our 3,000 patient records

numeric_cols = [
    'Age',
    'Length of Stay',
    'Number of Diagnoses',
    'Blood Pressure',
    'Blood Sugar Levels',
    'Previous Admissions'
]

# Generate descriptive statistics

display(Patients[numeric_cols].describe().T)

# COMMAND ----------

# Select numerical columns

numeric_cols = [
    'Age',
    'Length of Stay',
    'Number of Diagnoses',
    'Blood Pressure',
    'Blood Sugar Levels',
    'Previous Admissions'
]

# Create histograms for each numerical column
import seaborn as sns

for column in numeric_cols:

    plt.figure(figsize=(7, 4))

    sns.histplot(data=Patients, x=column, bins=15, kde=True)

    plt.title(f'Distribution of {column}')
    plt.xlabel(column)
    plt.ylabel('Number of Patients')

    plt.show()

# COMMAND ----------

# Count patients by admission type,this falls under categorical variables

admission_counts = pd.Series({
    'Elective': Patients['AdmissionType_Elective'].sum(),
    'Emergency': Patients['AdmissionType_Emergency'].sum()
})

display(admission_counts)

# COMMAND ----------

# Percentage of patients readmitted in each group (e.g. by Admission Type)

import pandas as pd
import matplotlib.pyplot as plt

# Admission Type was one-hot encoded earlier, so rebuild the group labels from the encoded columns
admission_type = Patients[['AdmissionType_Elective', 'AdmissionType_Emergency']].idxmax(axis=1).str.replace('AdmissionType_', '', regex=False)

# Build a crosstab of Readmission (Yes/No) per Admission Type
readmit_by_group = pd.crosstab(
    admission_type,
    Patients['Readmission'],
    normalize='index'
) * 100

# Show the percentage readmitted in each group
readmit_pct = readmit_by_group['Yes'].sort_values(ascending=False)
print('Percentage of patients readmitted in each group:')
display(readmit_pct)

# Plot the results as a bar chart
plt.figure(figsize=(7, 5))
readmit_pct.plot(kind='bar', color=['steelblue', 'salmon'])
plt.title('Readmission Rate by Group')
plt.xlabel('Group')
plt.ylabel('Readmitted (%)')
plt.xticks(rotation=0)
plt.ylim(0, 100)

# Show the percentage value on top of each bar
for index, value in enumerate(readmit_pct):
    plt.text(index, value + 2, f'{value:.1f}%', ha='center')

plt.show()

# COMMAND ----------

# Boxplots of numerical features by readmission outcome
# This shows how each variable is distributed for readmitted vs non-readmitted patients

import seaborn as sns
import matplotlib.pyplot as plt

numeric_cols = [
    'Age',
    'Length of Stay',
    'Number of Diagnoses',
    'Blood Pressure',
    'Blood Sugar Levels',
    'Previous Admissions'
]

for column in numeric_cols:
    plt.figure(figsize=(7, 4))
    sns.boxplot(data=Patients, x='Readmission', y=column)
    plt.title(f'{column} by Readmission Outcome')
    plt.xlabel('Readmission')
    plt.ylabel(column)
    plt.show()


# COMMAND ----------

# Correlation heatmap of the numerical features
# This shows how strongly each pair of variables is related (values range from -1 to +1)
# Values close to +1 or -1 mean a strong relationship; values close to 0 mean little or no relationship

import seaborn as sns
import matplotlib.pyplot as plt

# Calculate the correlation matrix for the numerical columns
numeric_cols = [
    'Age',
    'Length of Stay',
    'Number of Diagnoses',
    'Blood Pressure',
    'Blood Sugar Levels',
    'Previous Admissions'
]

correlation_matrix = Patients[numeric_cols].corr()

# Plot the heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Heatmap of Numerical Features')
plt.show()

# Summary: the correlations between the numerical features are weak (close to 0),
# which means none of the variables are strongly related to each other.
# This is good for modelling because the features each add independent information.
print(correlation_matrix)


# COMMAND ----------

# MAGIC %md
# MAGIC *Task 3: Feature Selection*

# COMMAND ----------

#In this section I want to identify which columns are useful for predicting whether a patient will be readmitted within 30 days.
print(Patients["Readmission_encoded"].value_counts())


# COMMAND ----------

# Target variable
y = Patients["Readmission_encoded"]

# Predictor variables
X = Patients[
    [
        "Age",
        "Gender_encoded",
        "Length of Stay",
        "Number of Diagnoses",
        "Blood Pressure",
        "Blood Sugar Levels",
        "Previous Admissions",
        "AdmissionType_Elective",
        "AdmissionType_Emergency"
    ]
]

print(X.head())
print(X.shape)

# COMMAND ----------

#A chi-square(χ²) test is a statistical hypothesis test used to determine whether there is a significant difference between observed data and expected data in categorical variables.Is Gender and Admission Type associated with patient readmission?
from scipy.stats import chi2_contingency

# Gender and Readmission
gender_table = pd.crosstab(
    Patients["Gender_encoded"],
    Patients["Readmission_encoded"]
)

chi2_gender, p_gender, dof_gender, expected_gender = chi2_contingency(gender_table)

print("Gender Chi-square:", chi2_gender)
print("Gender p-value:", p_gender) #In feature selection, a p-value helps determine whether the relationship between an input feature and the target variable is real or simply due to random chance. 

# COMMAND ----------

#  Admission Type column for the chi-square test
Patients["Admission_Type"] = Patients["AdmissionType_Elective"].map({
    1: "Elective",
    0: "Emergency"
})

# Admission Type and Readmission
admission_table = pd.crosstab(
    Patients["Admission_Type"],
    Patients["Readmission_encoded"]
)

chi2_admission, p_admission, dof_admission, expected_admission = chi2_contingency(admission_table)

print("Admission Type Chi-square:", chi2_admission)
print("Admission Type p-value:", p_admission)

# COMMAND ----------

#Gender has a very weak association with readmission.Admission Type also has a very weak association with readmission. Both values are close to zero, suggesting these features have little individual association with the target in this dataset.
import numpy as np

def cramers_v(chi2, table):
    n = table.to_numpy().sum()
    rows, columns = table.shape

    return np.sqrt(chi2 / (n * min(rows - 1, columns - 1)))


# Gender effect size
v_gender = cramers_v(chi2_gender, gender_table)
print("Gender Cramer's V:", v_gender)


# Admission Type effect size
v_admission = cramers_v(chi2_admission, admission_table)
print("Admission Type Cramer's V:", v_admission)

# COMMAND ----------

# DBTITLE 1,Mann-Whitney U test for numeric features
##We want to find out whether readmitted patients have different numerical characteristics from patients who were not readmitted.We will use the Mann–Whitney U test because it does not require the data to follow a normal distribution. We will also calculate an effect size.

from scipy.stats import mannwhitneyu

# Split patients into readmitted and not readmitted groups
readmitted = Patients[Patients["Readmission"] == "Yes"]
not_readmitted = Patients[Patients["Readmission"] == "No"]

# Numerical features to compare between the two groups
numeric_features = [
    "Age",
    "Length of Stay",
    "Number of Diagnoses",
    "Blood Pressure",
    "Blood Sugar Levels",
    "Previous Admissions"
]

results = []

for column in numeric_features:

    # Compare the two groups
    stat, p_value = mannwhitneyu(
        readmitted[column],
        not_readmitted[column],
        alternative="two-sided"
    )

    # Calculate effect size
    n1 = len(readmitted)
    n0 = len(not_readmitted)

    effect_size = 1 - (2 * stat) / (n1 * n0)

    results.append({
        "Feature": column,
        "U Statistic": stat,
        "P-value": p_value,
        "Effect Size": effect_size
    })

# Display results
numeric_results = pd.DataFrame(results)

display(numeric_results)

# COMMAND ----------

# MAGIC %md
# MAGIC *The Mann–Whitney U test was used to compare numerical features between readmitted and non-readmitted patients. All six features had p-values greater than 0.05, indicating no statistically significant differences between the groups. The effect sizes were also close to zero, suggesting very small differences. However, these features will not automatically be excluded because they may still contribute to predictions when combined with other variables.*

# COMMAND ----------

#Mutual Information (MI)helps us identify relationships between predictors and readmission, including relationships that may not be linear. Close to 0 means Little information about the target, Higher score means More information about the target

from sklearn.feature_selection import mutual_info_classif

# Calculate mutual information
mi_scores = mutual_info_classif(
    X,
    y,
    discrete_features=[
        False,  # Age
        True,   # Gender_encoded
        False,  # Length of Stay
        False,  # Number of Diagnoses
        False,  # Blood Pressure
        False,  # Blood Sugar Levels
        False,  # Previous Admissions
        True,   # AdmissionType_Elective
        True    # AdmissionType_Emergency
    ],
    random_state=42
)

# Create a results table
mi_results = pd.DataFrame({
    "Feature": X.columns,
    "Mutual Information": mi_scores
})

# Sort from highest to lowest
mi_results = mi_results.sort_values(
    by="Mutual Information",
    ascending=False
)

display(mi_results)

# COMMAND ----------

# MAGIC %md
# MAGIC *this means 1.Blood Sugar Levels	0.00700	Highest information score
# MAGIC 2.Previous Admissions	0.00434	Second highest
# MAGIC 3.Length of Stay	0.00238	Third highest*

# COMMAND ----------

# DBTITLE 1,Feature importance
#Feature importance
from sklearn.ensemble import RandomForestClassifier

# Train a Random Forest model on the selected features
# class_weight='balanced' handles the class imbalance noted earlier (more No than Yes outcomes)
rf_model = RandomForestClassifier(
    n_estimators=100,
    class_weight='balanced',
    random_state=42
)
rf_model.fit(X, y)

# Get feature importance scores
importance = rf_model.feature_importances_

# Create a results table
rf_results = pd.DataFrame({
    "Feature": X.columns,
    "Importance": importance
})

# Sort from highest to lowest
rf_results = rf_results.sort_values(
    by="Importance",
    ascending=False
)

display(rf_results)

# COMMAND ----------

# MAGIC %md
# MAGIC *The importance scores show how much each feature contributes to the Random Forest's decisions. Higher scores indicate greater importance within this model.*

# COMMAND ----------

# DBTITLE 1,Permutation importance
from sklearn.inspection import permutation_importance
# Calculate permutation importance
# The model was trained on X and y, so permutation importance is measured on the same features
perm_result = permutation_importance(
    rf_model,
    X,
    y,
    n_repeats=10,
    random_state=42,
    scoring="roc_auc"
)

# Create a results table
perm_results = pd.DataFrame({
    "Feature": X.columns,
    "Importance": perm_result.importances_mean
})

# Sort from highest to lowest
perm_results = perm_results.sort_values(
    by="Importance",
    ascending=False
)

display(perm_results)

# COMMAND ----------

# MAGIC %md
# MAGIC *Using a diferent methodology/approuch to assess feature importance by shuffling the values and measuring how the model accuracy changes,so if the accurancy drops significantly te feature is important*

# COMMAND ----------

# DBTITLE 1,Dropping Redundant Columns
# Drop the three features we do not need
X = X.drop(columns=[
    "Gender_encoded",
    "AdmissionType_Elective",
    "AdmissionType_Emergency"
])

# Check the remaining features
print(X.columns)

# Check the new dataset shape
print(X.shape)

# COMMAND ----------

# MAGIC %md
# MAGIC *Task 4:Model Building*

# COMMAND ----------

# DBTITLE 1,Training and Validation Data Preps
#A DummyClassifier is a simple model that makes predictions without learning patterns from patient data.We use it as a starting point to see whether our actual machine-learning models perform better.NB:This means the model will always predict the most common outcome.2134 patients were not readmitted(0)866 patients readmitted(1)

from sklearn.model_selection import train_test_split

# Set 20% for final testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# Split the remaining data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.25, random_state=42, stratify=y_train
)

print("Training data:", X_train.shape)
print("Validation data:", X_val.shape)
print("Testing data:", X_test.shape)

# COMMAND ----------

# DBTITLE 1,Building a DummyClassifier
#Build the DummyClassifier thruogh sklearn liabraries
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score

# dummyclassifier(strategy)this is going to predict the most commmon class
baseline_model = DummyClassifier(strategy="most_frequent")

# Trains the baseline using the training dat
baseline_model.fit(X_train, y_train)

# Make predictions on validation data
baseline_predictions = baseline_model.predict(X_val)

# Calculate how many predictions were correct
baseline_accuracy = accuracy_score(y_val, baseline_predictions)

print("Baseline Accuracy:", baseline_accuracy)

# COMMAND ----------

# DBTITLE 1,Building  Logistic Regression Model
#Logistic Regression is a classification algorithm. It estimates the probability that a patient will be readmitted within 30 days.I'll use class_weight="balanced" to help the model account for the smaller number of readmitted patients.

#Import the tools
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

# COMMAND ----------

# DBTITLE 1,Create and train the Model
# Create the Logistic Regression model
log_model = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(class_weight="balanced", max_iter=1000))
])

# Train the model
log_model.fit(X_train, y_train)

# COMMAND ----------

# MAGIC %md
# MAGIC *what the above code does,is to creates a classification Model via Logistic Regression(),using a aPipline that combines from scalling.Standard Scaler()scales the features using the training data.The weight balances the two outcomes,max_iter allows enough iteration for the model coverage.fit()obviously trains the model.*

# COMMAND ----------

# DBTITLE 1,Making Prediction
#y_pred gives the predicted class 0 or 1 and y_prob will give out probability of target(1)
# Predict readmission on validation data
y_pred = log_model.predict(X_val)

# Predict readmission probabilities
y_prob = log_model.predict_proba(X_val)[:, 1]

# COMMAND ----------

# DBTITLE 1,Evaluating  metrix on LogisticR.M
#These metrics help us understand different aspects of the model
print("Logistic Regression Results")

print("Accuracy:", accuracy_score(y_val, y_pred))
print("Precision:", precision_score(y_val, y_pred))
print("Recall:", recall_score(y_val, y_pred))
print("F1 Score:", f1_score(y_val, y_pred))
print("ROC-AUC:", roc_auc_score(y_val, y_prob))

# COMMAND ----------

# DBTITLE 1,Confusion Matrix
print(confusion_matrix(y_val, y_pred))

# COMMAND ----------

# MAGIC %md
# MAGIC *this shows True Negative=correctlt predicted No Admissions 211,
# MAGIC False Positives=predicted readmissions but the patients was not readmitted 92
# MAGIC False Negatives=missed patients who were readmitted 216
# MAGIC True Positives=correctly readamitted patients 81*
# MAGIC

# COMMAND ----------

# DBTITLE 1,RandomFrorestClassifier Vs the two Models
print("Best Random Forest settings:")
print(rf_grid.best_params_)

print("Best Cross-Validation ROC-AUC:")
print(rf_grid.best_score_)

# COMMAND ----------

# DBTITLE 1,Checking for the best Model
# Get the best models
best_log_model = log_grid.best_estimator_
best_tree_model = tree_grid.best_estimator_

# Make validation predictions
log_predictions = best_log_model.predict(X_val)
tree_predictions = best_tree_model.predict(X_val)

# Calculate ROC-AUC
log_auc = roc_auc_score(
    y_val, best_log_model.predict_proba(X_val)[:, 1]
)

tree_auc = roc_auc_score(
    y_val, best_tree_model.predict_proba(X_val)[:, 1]
)

print("Tuned Logistic Regression ROC-AUC:", log_auc)
print("Tuned Decision Tree ROC-AUC:", tree_auc)

# COMMAND ----------

# DBTITLE 1,Is The DT model Overfitting?
# Get the best Decision Tree model
best_tree_model = tree_grid.best_estimator_

# Predict probabilities on training data
tree_train_prob = best_tree_model.predict_proba(X_train)[:, 1]

# Predict probabilities on validation data
tree_val_prob = best_tree_model.predict_proba(X_val)[:, 1]

# Calculate ROC-AUC scores
from sklearn.metrics import roc_auc_score

train_auc = roc_auc_score(y_train, tree_train_prob)
val_auc = roc_auc_score(y_val, tree_val_prob)

print("Decision Tree Training ROC-AUC:", train_auc) #ReciverOperatingCharacteristcs
print("Decision Tree Validation ROC-AUC:", val_auc) #thePower it has on learning parttens on new data

# COMMAND ----------

# MAGIC %md
# MAGIC *The model performs much better on training data than on unseen validation data.i.e
# MAGIC 1.Training score (0.8365): The model learned patterns from the training data very well.
# MAGIC 2.Validation score (0.5365): The model struggles to recognise patterns in new data.
# MAGIC 3.The large gap suggests overfitting.
# MAGIC This means the Decision Tree may have memorised patterns in the training data instead of learning patterns that generalise well.*

# COMMAND ----------

# DBTITLE 1,Is RFC also Overfitting?
# Get the best Random Forest model
best_rf_model = rf_grid.best_estimator_

# Predict probabilities on training data
rf_train_prob = best_rf_model.predict_proba(X_train)[:, 1]

# Predict probabilities on validation data
rf_val_prob = best_rf_model.predict_proba(X_val)[:, 1]

# Calculate ROC-AUC scores
rf_train_auc = roc_auc_score(y_train, rf_train_prob)
rf_val_auc = roc_auc_score(y_val, rf_val_prob)

print("Random Forest Training ROC-AUC:", rf_train_auc)
print("Random Forest Validation ROC-AUC:", rf_val_auc)

# COMMAND ----------

# MAGIC %md
# MAGIC *Decision Tree: Performs well on training data but poorly on validation data.*

# COMMAND ----------

# DBTITLE 1,Hypertune on Logistic Regression
from sklearn.model_selection import GridSearchCV, StratifiedKFold

# Set up 5-fold cross-validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Choose values to test
log_param_grid = {
    "model__C": [0.01, 0.1, 1, 10, 100]
}

# Create GridSearchCV
log_grid = GridSearchCV(
    estimator=log_model,
    param_grid=log_param_grid,
    scoring="roc_auc",
    cv=cv,
    n_jobs=-1
)

# Train and test different combinations
log_grid.fit(X_train, y_train)

# Display the best settings and score
print("Best Logistic Regression settings:")
print(log_grid.best_params_)

print("Best Cross-Validation ROC-AUC:")
print(log_grid.best_score_)

# COMMAND ----------

# DBTITLE 1,Hypertune on Decision Tree
from sklearn.tree import DecisionTreeClassifier

# Create the Decision Tree
tree_model = DecisionTreeClassifier(
    class_weight="balanced",
    random_state=42
)

# Choose values to test
tree_param_grid = {
    "max_depth": [3, 5, 10, None],
    "min_samples_leaf": [5, 10, 20],
    "min_samples_split": [2, 10, 20]
}

# Create GridSearchCV
tree_grid = GridSearchCV(
    estimator=tree_model,
    param_grid=tree_param_grid,
    scoring="roc_auc",
    cv=cv,
    n_jobs=-1
)

# Train and test different combinations
tree_grid.fit(X_train, y_train)

# Display the best settings and score
print("Best Decision Tree settings:")
print(tree_grid.best_params_)

print("Best Cross-Validation ROC-AUC:")
print(tree_grid.best_score_)

# COMMAND ----------

# DBTITLE 1,Hypertune on Random Forest
#importing liabraries
from sklearn.ensemble import RandomForestClassifier
#create a model with rfc
# Create Random Forest model
rf_model = RandomForestClassifier(
    class_weight="balanced",
    random_state=42
)
# Choose values to test
rf_param_grid = {
    "n_estimators": [100, 200, 300], #number of decision trees in the foresst
    "max_depth": [5, 10, 20], #maximum depth of each tree
    "min_samples_leaf": [5, 10] #minimum patients allowed in each leaf
}
# Create GridSearchCV
rf_grid = GridSearchCV(
    estimator=rf_model,
    param_grid=rf_param_grid,
    scoring="roc_auc",
    cv=cv,
    n_jobs=-1
)

# Train and test different combinations
rf_grid.fit(X_train, y_train)
# Get the best Random Forest model
best_rf_model = rf_grid.best_estimator_

# Predict readmission
rf_predictions = best_rf_model.predict(X_val)

# Predict readmission probabilities
rf_probabilities = best_rf_model.predict_proba(X_val)[:, 1]

# Calculate ROC-AUC
rf_auc = roc_auc_score(y_val, rf_probabilities)

print("Random Forest Validation ROC-AUC:", rf_auc)

# COMMAND ----------

# Get the best Logistic Regression model
best_log_model = log_grid.best_estimator_

# Predict probabilities on training data
log_train_prob = best_log_model.predict_proba(X_train)[:, 1]

# Calculate training ROC-AUC
log_train_auc = roc_auc_score(y_train, log_train_prob)

print("Logistic Regression Training ROC-AUC:", log_train_auc)

# COMMAND ----------

# DBTITLE 1,DummyClassifer Model
#Starting off with a baseline;this is just a simple model that gives us a starting point for comparison on the three selected models.
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score

# Create the baseline model
baseline_model = DummyClassifier(strategy="most_frequent")

# Train the baseline model
baseline_model.fit(X_train, y_train)

# Make predictions
baseline_predictions = baseline_model.predict(X_val)

# Calculate accuracy
baseline_accuracy = accuracy_score(y_val, baseline_predictions)

print("Baseline Accuracy:", baseline_accuracy)

# COMMAND ----------

# DBTITLE 1,Training3Candidate Models
#1.Logistic Regression
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

log_model = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(
        class_weight="balanced",
        max_iter=1000 #the number of time the model enough attempts to find solutions
    ))
])

log_model.fit(X_train, y_train)

# COMMAND ----------

# DBTITLE 1,Training 3candidates Model
#2.Decision Tree
from sklearn.tree import DecisionTreeClassifier

tree_model = DecisionTreeClassifier(
    class_weight="balanced",
    random_state=42
)

tree_model.fit(X_train, y_train)

# COMMAND ----------

# DBTITLE 1,Training 3candidates Model
#3.Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier

rf_model = RandomForestClassifier(
    class_weight="balanced",
    random_state=42
)

rf_model.fit(X_train, y_train)

# COMMAND ----------

# MAGIC %md
# MAGIC /Comparing them on the same footing means that all three models must use:
# MAGIC 1. The same training data.
# MAGIC 2. The same validation data.
# MAGIC 3. The same target variable.
# MAGIC 4. The same evaluation metrics.NB:to make the comparison fair/

# COMMAND ----------

# DBTITLE 1,Handling Class Imbalance
#3.Class_weight="balanced" It gives more importance to the minority class (Readmission = Yes) during training.This will also mean the No Readmissions will receives less weight. And Readmissions Yes: will receives more weight.This encourages the model to pay more attention to readmitted patients.

# COMMAND ----------

# DBTITLE 1,Cross-Validation
#This is the stage where we check how consistently the three models perform on different portions of the training data,it is then divided into 5parts/folds
#4 parts/folds for training and 1 fold is for validation.This process is repeated 5times with 1 different fold used in validation

# Comparing the cross-validation results

models = {
    "Logistic Regression": log_grid,
    "Decision Tree": tree_grid,
    "Random Forest": rf_grid
}

for name, grid in models.items():

    # Get the best model's cross-validation results
    best_index = grid.best_index_

    mean_score = grid.cv_results_["mean_test_score"][best_index]
    std_score = grid.cv_results_["std_test_score"][best_index]

    print(name)
    print("Mean ROC-AUC:", round(mean_score, 4)) #ability to know yes and no on addmission
    print("Standard Deviation:", round(std_score, 4)) #the scores diffrence on different 5 folds
    print("----------------------")

# COMMAND ----------

# MAGIC %md
# MAGIC *Stratified 5-fold cross-validation was used to compare Logistic Regression, Decision Tree and Random Forest. ROC-AUC was selected because it measures the model's ability to distinguish between the two readmission classes without relying on a fixed prediction threshold. The mean ROC-AUC and standard deviation were used to assess model performance and consistency across folds.*

# COMMAND ----------

# DBTITLE 1,Reducing overfitting on DT
tree_param_grid2 = {
    "max_depth": [2, 3, 4, 5],
    "min_samples_leaf": [10, 20, 30, 50],
    "min_samples_split": [10, 20, 30]
}
# the second hyperparameter search

tree_grid2 = GridSearchCV(
    tree_model,
    tree_param_grid2,
    scoring="roc_auc",
    cv=cv,
    n_jobs=-1
)

tree_grid2.fit(X_train, y_train)

# Display the results
print("Best Parameters:", tree_grid2.best_params_)
print("Best CV ROC-AUC:", tree_grid2.best_score_)

# COMMAND ----------

# MAGIC %md
# MAGIC *The Decision Tree was provisionally selected as the final candidate because it achieved the highest mean cross-validation ROC-AUC of 0.5280 and validation ROC-AUC of 0.5365 among the three models. It also offers better interpretability than Random Forest, allowing healthcare professionals to understand the decision-making process. Although Random Forest showed greater stability across folds, its predictive performance was lower and its training score indicated substantial overfitting. The Decision Tree also showed signs of overfitting, so further evaluation is necessary. Since the hospital needs reliable identification of patients at risk of readmission, recall, false negatives and model reliability must also be considered before deployment.*

# COMMAND ----------

# MAGIC %md
# MAGIC #Task 5: Model Evaluation

# COMMAND ----------

#In this step,we are about to find out how well the model performs on patients it has never seen during training or validation.
#Step 1: Prepare the Final model

# Select the best Decision Tree model

final_model = tree_grid.best_estimator_

# COMMAND ----------

#Step 2: we make predictions on  readmission using the test data

y_test_pred = final_model.predict(X_test)

# COMMAND ----------

#Step 3: Calculating the evalution metrics
from sklearn.metrics import accuracy_score, precision_score
from sklearn.metrics import recall_score, f1_score, roc_auc_score

# Calculate evaluation metrics

accuracy = accuracy_score(y_test, y_test_pred)
precision = precision_score(y_test, y_test_pred)
recall = recall_score(y_test, y_test_pred)
f1 = f1_score(y_test, y_test_pred)

# Predict probabilities for ROC-AUC
y_test_prob = final_model.predict_proba(X_test)[:, 1]

roc_auc = roc_auc_score(y_test, y_test_prob)

# Display results
print("Accuracy:", round(accuracy, 4)) #How many predictions were correct overall?
print("Precision:", round(precision, 4))#for predicted readmissions,how many were true?
print("Recall:", round(recall, 4))#of patients actually readmitted,how many did the model identify?
print("F1-score:", round(f1, 4))#balacing between precision and recall
print("ROC-AUC:", round(roc_auc, 4))#How well can the model distingush between the two classes?

# COMMAND ----------

# MAGIC %md
# MAGIC Metric	Result	Interpretation
# MAGIC Accuracy	50.33%	About half of predictions were correct.
# MAGIC Precision	27.44%	About 27 out of 100 predicted readmissions were correct.
# MAGIC Recall	43.93%	The model identified about 44 out of 100 actual readmissions.
# MAGIC F1-score	33.78%	Weak balance between precision and recall.
# MAGIC ROC-AUC	0.481	Poor ability to distinguish the two classes.

# COMMAND ----------

#Step 3: Confusion Matrix


# COMMAND ----------

# MAGIC %md
# MAGIC The confusion matrix showed 226 true negatives, 201 false positives, 97 false negatives and 76 true positives. The model correctly identified 76 readmitted patients but missed 97 actual readmissions. The high number of false negatives is concerning because patients who may require additional follow-up care could be overlooked. The false positives could also increase the hospital's workload through unnecessary follow-up assessments. These results indicate that the Decision Tree requires improvement before it can be considered for hospital use.

# COMMAND ----------

# DBTITLE 1,Evaluating Roc curve
#ROC Curve:This will show how well the model separates patients who are readmitted from those who are not.Also The ROC curve compares:
#True Positive Rate (TPR): the proportion of actual readmissions correctly identified(226).False Positive Rate (FPR): the proportion of non-readmissions incorrectly predicted as readmissions.(201)

from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt

# Calculate ROC curve values
fpr, tpr, thresholds = roc_curve(y_test, y_test_prob)

# Calculate ROC-AUC
roc_auc = roc_auc_score(y_test, y_test_prob)

# Plot ROC curve
plt.figure(figsize=(7, 5))
plt.plot(fpr, tpr, label=f"Decision Tree (AUC = {roc_auc:.3f})")

# Add random guessing line
plt.plot([0, 1], [0, 1], linestyle="--", label="Random Guessing")

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Decision Tree ROC Curve")
plt.legend()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC  The Decision Tree performs poorly at distinguishing between readmitted and non-readmitted patients. Its ROC curve is very close to the random guessing line.

# COMMAND ----------

# DBTITLE 1,Evaluating Precision Recall Curve
#The Precision-Recall curve focuses on the positive class: patients who were readmitted.Recall: Of all patients actually readmitted, how many did the model identify?

from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score

# Calculate Precision-Recall values
precision_values, recall_values, thresholds = precision_recall_curve(
    y_test, y_test_prob
)

# Calculate Average Precision
avg_precision = average_precision_score(y_test, y_test_prob)

# Plot Precision-Recall curve
plt.figure(figsize=(7, 5))

plt.plot(
    recall_values,
    precision_values,
    label=f"Decision Tree (AP = {avg_precision:.3f})"
)

# Add baseline
baseline = y_test.mean()
plt.axhline(y=baseline, linestyle="--", label="Baseline")

plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Decision Tree Precision-Recall Curve")
plt.legend()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Overfitting Mitigation, Limitations and Ethical Considerations
# MAGIC
# MAGIC **Overfitting Mitigation**
# MAGIC
# MAGIC Several techniques were applied to reduce overfitting, including feature selection, hyperparameter tuning and 5-fold stratified cross-validation. Model complexity was also controlled by adjusting Decision Tree parameters such as maximum depth, minimum samples per leaf and minimum samples required to split a node.
# MAGIC
# MAGIC Despite these measures, the Decision Tree still showed signs of overfitting, with a training ROC-AUC of 0.8365 compared to a validation ROC-AUC of 0.5365. Its final test ROC-AUC was 0.481, indicating poor generalisation to unseen data.
# MAGIC
# MAGIC **Limitations**
# MAGIC
# MAGIC - The dataset contains only 3,000 patient records, which may limit the model's ability to learn complex patterns.
# MAGIC - The target variable is imbalanced, with fewer readmitted patients than non-readmitted patients.
# MAGIC - The final model achieved weak predictive performance, with a ROC-AUC of 0.481.
# MAGIC - The confusion matrix showed 97 false negatives, meaning the model failed to identify 97 patients who were actually readmitted.
# MAGIC - The dataset contains a limited number of clinical features, and important factors influencing readmission may be missing.
# MAGIC - The model has not been externally validated using data from other hospitals.
# MAGIC
# MAGIC **Ethical Considerations**
# MAGIC
# MAGIC Patient privacy and confidentiality must be protected when handling healthcare data. The model may also introduce bias if certain patient groups are underrepresented or if their outcomes are predicted less accurately.
# MAGIC
# MAGIC False negative predictions could create patient safety risks if healthcare professionals rely on the model without further assessment. Therefore, predictions should only support clinical decision-making and should never replace professional medical judgement.
# MAGIC
# MAGIC The model should not be deployed in a real healthcare environment without further improvement, independent validation, fairness assessment and clinical review.

# COMMAND ----------

# MAGIC %md
# MAGIC #checking if the model will perform better after feature reductions of NUmber of Diagnoses and Previous readmission

# COMMAND ----------

#Step 1,creating a reduced dataset
