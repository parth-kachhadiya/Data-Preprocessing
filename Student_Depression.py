"""
Gender : <class 'str'>
Age : <class 'float'> 
City : <class 'str'>
Profession : <class 'str'>
Academic Pressure : <class 'float'> 
Work Pressure : <class 'float'> 
CGPA : <class 'float'> 
Study Satisfaction : <class 'float'> 
Job Satisfaction : <class 'float'> 
Sleep Duration : <class 'str'>
Dietary Habits : <class 'str'>
Degree : <class 'str'>
Have you ever had suicidal thoughts ? : <class 'str'>
Work/Study Hours : <class 'float'> 
Financial Stress : <class 'float'> 
Family History of Mental Illness : <class 'str'>
Depression : <class 'int'> (O/p)
"""

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score


def remove_outliers(dataframe : pd.DataFrame, target_columns : list) -> pd.DataFrame :

    temp = dataframe.copy()

    for cols in target_columns:
        Q1 = temp[cols].quantile(0.25)
        Q3 = temp[cols].quantile(0.75)
        
        IQR = Q3 - Q1
        
        lower_limit = Q1 - 1.5 * IQR
        upper_limit = Q3 + 1.5 * IQR

        # replace the outliers with lower or upper bound
        temp[cols] = np.where(
            temp[cols] < lower_limit, 
            lower_limit,
            np.where(
                temp[cols] > upper_limit, 
                upper_limit,
                temp[cols]
            )
        )
    return temp

def func(pct, allvalues):
    # Handle NaN values in allvalues by replacing them with 0
    allvalues = [value if not pd.isna(value) else 0 for value in allvalues]
    
    # Calculate the absolute count from the percentage, avoiding NaN errors
    absolute = int(pct / 100. * sum(allvalues)) if sum(allvalues) > 0 else 0
    return f"{absolute}\n({pct:.1f}%)"

def generate_report(dataset : pd.DataFrame, basic_type : dict):
    for_numerical = dataset.select_dtypes(include='number').describe().T
    for_categorical = dataset.select_dtypes(include='object').describe().T

    for_numerical["NaNs"] = dataset.select_dtypes(include='number').isnull().sum()
    for_numerical["NaN_per"] = (dataset.select_dtypes(include='number').isnull().sum() / dataset.shape[0]) * 100
    for_numerical["Uniques"] = dataset.select_dtypes(include='number').nunique()
    for_numerical["Unique_per"] = (dataset.select_dtypes(include='number').nunique() / dataset.shape[0]) * 100

    for cols in for_numerical.index:
        for_numerical.at[cols, "BaseType"] = basic_type[cols]

    return (for_numerical, for_categorical)

def get_type_of_each_row(dataframe : pd.DataFrame):
    datatypes = {}
    for column in dataframe.columns:
        sub_values = []
        for values in dataframe[column]:
            sub_values.append(str(type(values)))
        datatypes[column] = max(sub_values)
        sub_values.clear()
    return datatypes

def categorical_data_analysis(dataframe : pd.DataFrame,dataset : pd.DataFrame):
    for column in dataset.index:
        print(f"{column} : {dataframe[column].nunique()}, {dataframe[column].unique()}")
        print("*" * 10)

dataset = pd.read_csv("D:\\Machine_Learning\\codes\\Own_model\\data_csv\\Student_Depression_Dataset.csv").drop('id',axis=1)

report_num, report_cat = generate_report(dataset, get_type_of_each_row(dataset))

""" - <> - <> - <> - <> - <> - <> - <> - <> - <> - <> - <> - <> - EDA - <> - <> - <> - <> - <> - <> - <> - <> - <> - <> - <> - <> - <> - <> - """

# Data distribution using histogram..

for column in dataset.select_dtypes(include='number'):
    sns.histplot(data = dataset, x=column)
    plt.show()


# Outliers detection using boxplot..

for column in dataset.select_dtypes(include='number'):
    sns.boxplot(data = dataset, x=column)
    plt.show()


# Scatter plot to understand the relation between indipendent veriables..

for column in ['Age', 'Academic Pressure', 'Work Pressure', 'CGPA',
       'Study Satisfaction', 'Job Satisfaction', 'Work/Study Hours',
       'Financial Stress']:
    sns.scatterplot(data = dataset,x=column, y='Depression')
    plt.show()


""" > - <> - <> - <> - <> - <> - <> - <> - <> - <> - <> - < Age wise data analysis > - <> - <> - <> - <> - <> - <> - <> - <> - <> - < """


sns.lineplot(data = dataset, x = dataset['Age'], y = dataset['Academic Pressure'])
plt.show()
sns.lineplot(data = dataset, x = dataset['Age'], y = dataset['Work Pressure'])
plt.show()
sns.lineplot(data = dataset, x = dataset['Age'], y = dataset['Study Satisfaction'])
plt.show()
sns.lineplot(data = dataset, x = dataset['Age'], y = dataset['Job Satisfaction'])
plt.show()
sns.lineplot(data = dataset, x = dataset['Age'], y = dataset['Work/Study Hours'])
plt.show()
sns.lineplot(data = dataset, x = dataset['Age'], y = dataset['Financial Stress'])
plt.show()


""" > - <> - <> - <> - <> - <> - <> - <> - <> - <> - <> - < Profession wise data analysis > - <> - <> - <> - <> - <> - <> - <> - <> - <> - < """

group_of_profession = dataset.groupby("Profession")

p_group_of_age = group_of_profession['Age'].mean().reset_index()
p_group_of_ac_pre = group_of_profession['Academic Pressure'].mean().reset_index()
p_group_of_work_pre = group_of_profession['Work Pressure'].mean().reset_index()
p_group_of_study_satis = group_of_profession['Study Satisfaction'].mean().reset_index()
p_group_of_job_satis = group_of_profession['Job Satisfaction'].mean().reset_index()


sns.barplot(data=p_group_of_job_satis, x='Profession', y='Job Satisfaction')
sns.lineplot(data=p_group_of_job_satis, x='Profession', y='Job Satisfaction', color='red', lw=2, marker='o')

plt.xticks(rotation=90)
plt.tight_layout()
plt.show()

p_group_sleep_dur = group_of_profession['Sleep Duration'].value_counts().unstack()


for profession in ['Student', 'Civil Engineer', 'Architect', 'UX/UI Designer',
 'Digital Marketer', 'Content Writer', 'Educational Consultant', 'Teacher',
 'Manager' ,'Chef' ,'Doctor' ,'Lawyer' ,'Entrepreneur', 'Pharmacist']:
    plt.figure(figsize=(7, 7))
    p_group_sleep_dur.loc[profession].plot.pie(autopct=lambda pct: func(pct, p_group_sleep_dur.loc[profession]), startangle=90, labels=p_group_sleep_dur.columns, title=f'Sleep Duration Distribution for {profession}')
    plt.show()


""" > - <> - <> - <> - <> - <> - <> - <> - <> - <> - <> - < Gender wise data analysis > - <> - <> - <> - <> - <> - <> - <> - <> - <> - < """

''' Main grouped object '''
group_of_genders = dataset.groupby('Gender')

''' Category - 1 barplot'''
academic_pressure_value = group_of_genders['Academic Pressure'].mean().reset_index()
gender_value = group_of_genders['Age'].median().reset_index()
work_pressure_value = group_of_genders['Work Pressure'].mean().reset_index()
cgpa_value = group_of_genders['CGPA'].mean().reset_index()
study_satisfaction_value = group_of_genders['Study Satisfaction'].mean().reset_index()
job_satisfaction_value = group_of_genders['Job Satisfaction'].mean().reset_index()
work_study_hour_value = group_of_genders['Work/Study Hours'].mean().reset_index()
financial_stress = group_of_genders['Financial Stress'].mean().reset_index()

sns.barplot(data=financial_stress, x='Gender', y='Financial Stress')
plt.show()
''' Category - 2.1 Pie chart (Sleep duration) '''
sleep_duration = group_of_genders['Sleep Duration'].value_counts().unstack()
# Plot for Female
plt.figure(figsize=(7, 7))
sleep_duration.loc['Female'].plot.pie(autopct=lambda pct: func(pct, sleep_duration.loc['Female']), startangle=90, labels=sleep_duration.columns, title='Sleep Duration Distribution for Female')

# Plot for Male
plt.figure(figsize=(7, 7))
sleep_duration.loc['Male'].plot.pie(autopct=lambda pct: func(pct, sleep_duration.loc['Male']), startangle=90, labels=sleep_duration.columns, title='Sleep Duration Distribution for Male')
# plt.show()


''' Category - 2.2 Pie chart (Sucide thoughts) '''
sucide_thought_value = group_of_genders['Have you ever had suicidal thoughts ?'].value_counts().unstack()

plt.figure(figsize=(7, 7))
sucide_thought_value.loc['Female'].plot.pie(autopct=lambda pct: func(pct, sucide_thought_value.loc['Female']), startangle=90, labels=sucide_thought_value.columns, title='Sucide thoughs for Female')

plt.figure(figsize=(7, 7))
sucide_thought_value.loc['Male'].plot.pie(autopct=lambda pct: func(pct, sucide_thought_value.loc['Male']), startangle=90, labels=sucide_thought_value.columns, title='Sucide thoughs for Male')
# plt.show()

''' Category - 2.3 Pie chart (Depression) count '''
depression_counting = group_of_genders['Depression'].value_counts().unstack()

plt.figure(figsize=(7, 7))
depression_counting.loc['Female'].plot.pie(autopct=lambda pct: func(pct, depression_counting.loc['Female']), startangle=90, labels=depression_counting.columns, title='Depressed females')

plt.figure(figsize=(7, 7))
depression_counting.loc['Male'].plot.pie(autopct=lambda pct: func(pct, depression_counting.loc['Male']), startangle=90, labels=depression_counting.columns, title='Depressed males')
# plt.show()



dataset = remove_outliers(dataset, list(dataset.select_dtypes(include='number').columns)).drop(['Work Pressure','Job Satisfaction'],axis=1)
imputer = SimpleImputer(missing_values=np.nan,strategy='most_frequent')
dataset['Financial Stress'] = imputer.fit_transform(dataset[['Financial Stress']])

X = dataset.iloc[:,:-1]
Y = dataset.iloc[:,-1]

x_train, x_test, y_train, y_test = train_test_split(X,Y,test_size=.2)

sleep_order = ["Others","More than 8 hours","7-8 hours", "5-6 hours", "Less than 5 hours"]
diet_order = ["Others", "Healthy", "Moderate", "Unhealthy"]
pressure_order = [0. ,1. ,2. ,3., 4., 5.]
study_satis_order = [0. ,1. ,2. ,3., 4., 5.]
finance_stress_order = [1., 2., 3., 4., 5.]

transformer = ColumnTransformer(
    transformers=[
        ("feture_nominal_categories", OneHotEncoder(handle_unknown='ignore'), ['Gender','City','Profession','Family History of Mental Illness','Have you ever had suicidal thoughts ?','Work/Study Hours','Degree']),
        ("feture_ordinal_categories", OrdinalEncoder(categories=[sleep_order, diet_order, pressure_order, study_satis_order, finance_stress_order]), ['Sleep Duration','Dietary Habits','Academic Pressure','Study Satisfaction','Financial Stress']),
        ("feture_numeric_categories", StandardScaler(), ['Age','CGPA'])
    ],    
    remainder='passthrough'
)
transformer.fit(x_train)
x_train = transformer.transform(x_train)
x_test = transformer.transform(x_test)

model = RandomForestClassifier(random_state=42, max_depth=10)
model.fit(x_train,y_train)

y_hat = model.predict(x_test)

classi_report = classification_report(y_test, y_hat)
training_accuracy = accuracy_score(y_train, model.predict(x_train))
score = accuracy_score(y_test, y_hat)
cross_validation_score = cross_val_score(model, x_train, y_train, cv=5, scoring='accuracy')

print(cross_validation_score.mean())

feture_importance = model.feature_importances_
fetures = transformer.get_feature_names_out()
