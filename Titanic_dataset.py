import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score

from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

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

def identify_outliers(dataframe : pd.DataFrame):
    total_outliers = {}
    for cols in dataframe.select_dtypes(include='number').columns:
        Q1 = dataframe[cols].quantile(0.25)
        Q3 = dataframe[cols].quantile(0.75)
        
        IQR = Q3 - Q1
        
        lower_limit = Q1 - 1.5 * IQR
        upper_limit = Q3 + 1.5 * IQR

        total_outliers[cols] = f"Below lower limit(${lower_limit}) total '{dataframe[dataframe[cols] < lower_limit].shape[0]}' values, Above higher limit({upper_limit}) total '{dataframe[dataframe[cols] > upper_limit].shape[0]}' values"
    return total_outliers

def remove_outliers(dataframe : pd.DataFrame, target_columns : list) -> pd.DataFrame:
    temp = dataframe.copy()
    for cols in target_columns:
        Q1 = temp[cols].quantile(0.25)
        Q3 = temp[cols].quantile(0.75)
        
        IQR = Q3 - Q1
        
        lower_limit = Q1 - 1.5 * IQR
        upper_limit = Q3 + 1.5 * IQR

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

dataset = pd.read_csv("Titanic_dataset.csv").drop(["PassengerId","Cabin","Name","Ticket"],axis=1)

dataset['FamilyMembers'] = dataset['SibSp'] + dataset['Parch']
dataset.drop(['SibSp','Parch'],axis=1,inplace=True)

# Calculate survival ratio based on Pclass
survival_ratio = dataset.groupby('Pclass')['Survived'].mean().reset_index()

# Identifying outliers
total_outliers = identify_outliers(dataset)

# Removing outliers
dataset = remove_outliers(dataset, list(dataset.select_dtypes(include='number').columns))

report_num, report_cat = generate_report(dataset, get_type_of_each_row(dataset))

# Few survival related analysis
pickup_station_analysis = dataset.groupby(['Embarked','Survived'])['Survived'].value_counts().unstack()
sur_chance_on_pclass = dataset.groupby(['Pclass','Survived'])['Survived'].count().unstack()



sns.barplot(data=survival_ratio, x='Pclass', y='Survived')
plt.title('Survival Ratio by Passenger Class (Pclass)')
plt.xlabel('Passenger Class (Pclass)')
plt.ylabel('Survival Ratio')
plt.grid(color='gray', linestyle='--', linewidth=.5, alpha=.5)
plt.gca().set_facecolor("#f7f7f7")
plt.show()

mean_fare = dataset.groupby('FamilyMembers')['Fare'].mean().reset_index()
# Plotting a barplot for mean Fare by FamilyMembers
sns.barplot(data=mean_fare, x='FamilyMembers', y='Fare')
plt.title('Mean Fare by Number of Family Members')
plt.xlabel('Number of Family Members')
plt.ylabel('Mean Fare')
plt.grid(color='gray', linestyle='--', linewidth=.5, alpha=.5)
plt.gca().set_facecolor("#f7f7f7")
plt.show()

sns.scatterplot(data=dataset, x='FamilyMembers', y='Fare')
plt.title('Fare by Number of Family Members')
plt.xlabel('Number of Family Members')
plt.ylabel('Fare')
plt.grid(color='gray', linestyle='--', linewidth=.5, alpha=.5)
plt.gca().set_facecolor("#f7f7f7")
plt.show()

sns.countplot(data=dataset, x='Sex')
plt.title('Count of Passengers by Gender')
plt.xlabel('Gender')
plt.ylabel('Count')
plt.show()

sns.scatterplot(data=dataset, x='Age', y='Fare', hue='Sex', style='Survived')
plt.grid(color='gray', linestyle='--',linewidth=.5, alpha=.5)
plt.gca().set_facecolor("#f7f7f7")
plt.show()

correlation_matrix = dataset.corr(numeric_only=True)
sns.heatmap(correlation_matrix,annot=True)
plt.show()


Y = dataset.iloc[:,0]
X = dataset.iloc[:,1:]

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=.2, random_state=42)

imputer = SimpleImputer(strategy='mean')
x_train[['Age','Fare']] = imputer.fit_transform(x_train[['Age','Fare']])
x_test[['Age','Fare']] = imputer.transform(x_test[['Age','Fare']])

transformer = ColumnTransformer(
    [
        ("op_0", OrdinalEncoder(categories=[['Q','C','S'],[1,2,3]]), [4,0]),
        ("op_1", OneHotEncoder(drop='first',sparse_output=False,handle_unknown='ignore'), [1,5]),
        ("op_2", MinMaxScaler(), [2,3]),
    ],
    remainder='passthrough'
)

transformer.fit(x_train)
x_train = transformer.transform(x_train)
x_test = transformer.transform(x_test)

# Establishing pipeline

# Stage - 1 of pipeline (encoding categorical veriables)
transformer_1 = ColumnTransformer(
    [
        ("OHE_categories", OneHotEncoder(drop='first',sparse_output=False,handle_unknown='ignore'), [0,1,4,5])
    ],
    remainder='passthrough'
)

# Stage - 2 of pipeline (Scaling data on same scale through 'StandardScaler' or 'MinMaxScaler')
transformer_2 = ColumnTransformer(
    [
        ("SS_numerics", StandardScaler(), [2,3])
    ],  
    remainder='passthrough'
)

# Stage - 3 of pipeline (Model) OPTIONAL
model_1 = RandomForestClassifier(n_estimators=10, max_depth=2, random_state=42)
model_2 = DecisionTreeClassifier(max_depth=3, min_samples_split=10, random_state=42)

# Establishing pipeline connection
pipeline = Pipeline([
    ('DP_1', transformer_1),
    ('DP_2', transformer_2),
    ("Model_training", model_1)
])

'''
NOTE : Remember if you don't insert RandomForestClassifier as process of pipeline, 
        you don't call 'fit' method instead of you should use 'fit_transform' method

       Because of you have selected your model as last process of pipeline,
       your model will get the data and afterwards you will directly use 'predict' method.
'''
pipeline.fit(x_train, y_train)

# Extrecting information from process done by 'pipeline'

# 1st : getting all process information
for id, process in pipeline.named_steps.items():
    print(f"{id} : {process}")


# 2nd : Getting additional information about individuals (for debugging purpose)
print(pipeline.named_steps['DP_2'].transformers_)

# Prediction
y_hat = pipeline.predict(x_test)

# Train/Test score
training_accuracy = accuracy_score(y_train, pipeline.predict(x_train))
test_accuracy = accuracy_score(y_test, y_hat)
print(f"Training accuracy score: {training_accuracy}, Test accuracy score: {test_accuracy}")
