import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import KNNImputer

def generate_report(dataset : pd.DataFrame):
    '''
    This method will generate a brief report about dataset
    such as column name,mean, max, standard deviation, 'NaN' values, percentage of 'NaN' values and much more..
    '''

    repeating_rows = dataset.duplicated()
    report = dataset.describe().T

    report['DataType'] = dataset.columns.dtype
    report['NaN_values'] = dataset.isnull().sum()
    report['NaN_per'] = (dataset.isnull().sum() / dataset.shape[0]) * 100
    report['Unique'] = dataset.nunique()
    report['Unique_per'] = (dataset.nunique() / dataset.shape[0]) * 100

    return (report, repeating_rows[repeating_rows == True].index.tolist())

def wisker(column):
    Q1, Q3 = np.percentile(column, [25, 75])
    IQR = Q3 - Q1 # Inter Quartile Range

    # Setting boundries..
    lwr_wisker = Q1 - 1.5 * IQR
    upr_wisker = Q3 + 1.5 * IQR

    return (lwr_wisker, upr_wisker)


dataset = pd.read_csv("D:\\Machine_Learning\\codes\\Own_model\\data_csv\\Student_Performance.csv")

""" - <> - <> - <> - <> - <> - <> - <> - <> - <> - <> - <> - <> - Statistics - <> - <> - <> - <> - <> - <> - <> - <> - <> - <> - <> - <> - <> - <> - """
report, list_of_repeating_records = generate_report(dataset)

""" - <> - <> - <> - <> - <> - <> - <> - <> - <> - <> - <> - <> - EDA - <> - <> - <> - <> - <> - <> - <> - <> - <> - <> - <> - <> - <> - <> - """

# Histogram to understand the distribution of data

for column in dataset.select_dtypes(include='number').columns:
    sns.histplot(data=dataset, x=column)
    plt.show()


# Box-plot to identify outliers

for column in dataset.select_dtypes(include='number').columns:
    sns.boxplot(data=dataset,x=column)
    plt.show()


# Scatter plot to understand relationship between dependent and indipendent veriables

for column_name in ['Hours Studied', 'Previous Scores', 'Extracurricular Activities', 'Sleep Hours', 'Sample Question Papers Practiced']:
    sns.scatterplot(data=dataset, x=column_name,y='Performance Index')
    plt.show()


# correlation with heatmap to interpret the relation and multicolliniarity

correlation_matrix = dataset.select_dtypes(include='number').iloc[:,:-1].corr()
plt.figure(figsize=(10,10))
sns.heatmap(correlation_matrix, annot=True)
plt.show()


""" - <> - <> - <> - <> - <> - <> - <> - <> - <> - <> - <> - <> - Missing values - <> - <> - <> - <> - <> - <> - <> - <> - <> - <> - <> - <> - <> - <> - 

Possible options : Mean, Median(If outliers), Mode, KNNInputer
"""

# Method-1..

for columns in ['Hours Studied', 'Previous Scores', 'Sleep Hours', 'Sample Question Papers Practiced']:
    dataset[columns].fillna(dataset[columns].median(), inplace=True)


# Method-2..

imputer = KNNImputer() # Average of the nearest values
for columns in dataset.select_dtypes(include='number').columns:
    dataset[columns] = imputer.fit_transform(dataset[[columns]])


""" - <> - <> - <> - <> - <> - <> - <> - <> - <> - <> - <> - <> - Outliers - <> - <> - <> - <> - <> - <> - <> - <> - <> - <> - <> - <> - <> - <> - 

👉 Outlier treatment only applicable on continues numerical data
👉 We don't need to apply outliers treatment in output veriable
👉 Outliers treatment only accplicable on continues numerical data
👉 We can't apply outlier treatment on categorical or descreate veriables
for this sample dataset 'Previous scores' named column is only column which contains continues numerical values
"""

target_columns = ['Previous Scores']

for affected_column in target_columns:
    boundries = wisker(dataset[affected_column])
    dataset[affected_column] = np.where(dataset[affected_column] < boundries[0], boundries[0], dataset[affected_column])
    dataset[affected_column] = np.where(dataset[affected_column] > boundries[1], boundries[1], dataset[affected_column])



""" - <> - <> - <> - <> - <> - <> - <> - <> - <> - <> - <> - <> - Duplicate & Garbage values - <> - <> - <> - <> - <> - <> - <> - <> - <> - <> - <> - <> - <> - <> - """
dataset.drop_duplicates()

""" - <> - <> - <> - <> - <> - <> - <> - <> - <> - <> - <> - <> - Data encoding - <> - <> - <> - <> - <> - <> - <> - <> - <> - <> - <> - <> - <> - <> - 

Methods: 1. OneHotEncoding
         2. LabelEncoding
"""


dummy = pd.get_dummies(dataset,columns=["Extracurricular Activities"], drop_first=True)
