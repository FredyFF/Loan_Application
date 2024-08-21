# [Case Study] Loan Application Aproval 

### Table of Contents

- [Project Overview](#project-overview)
- [Data Sources](#data-sources)
- [Recommendations](#recommendations)

### Project Overview
---
This data analysis project aims to provide insights into how machine learning plays a role in making decisions in providing loans to customers. We will use Machine Learning with Python to facilitate the job of loan originators and predict whether candidate profiles are relevant or not using key features such as Status Marriage, Education, Applicant's Income, Credit History, etc.

### Data Sources

Loan Aproval: The primary dataset used for this analysis is the "Loan Aproval.csv" file : https://docs.google.com/spreadsheets/d/1i_VDs1HriRcv6JhuRwsqW4ElUiqLKHBilISfzCwxcxA/edit?gid=73737849#gid=73737849

### Tools

- GoogleSheet - Data Cleaning
- Jupyter NoteBook - Analysis and Report

### Data Cleaning/Preparation

In the initial data preparation phase, we performed the following tasks:
1. Data loading and inspection.
2. Handling missing values.
3. Data cleaning and formatting.

EDA involved exploring the loan aproval to answer key questions, such as:

- What is the overall shows which value dominates ?
- how each of these variables are related or influential to each other to produce a strong correlation
- Is there an ideal method to apply in classification to generate optimal loan distribution?

### Data Analysis
The dataset contains 13 features:

|Loan|Unique ID|
|--------|--------|
 | Gender | Applicant gender: Male/female | 
 | Marital status | For married applicants, the score is Yes/No | 
 | Dependents | Indicates whether the applicant has dependents or not. | 
 | Education | This will tell us whether the applicant has Passed or not | 
 | Work alone | It defines that the applicant is self-employed i.e. Yes/No | 
 | Income | Applicant Income Applicant | 
 | Applicant's Income | Joint Applicant's Income | 
 | Loan Amount | Loan Amount (in thousands) | 
 | Loan Amount Term | Loan term (in months) | 
 | Credit history | A person's debt repayment credit history | 
 | Property Area | The property area is Rural/Urban/Semi-urban | 
 | Loan Status | Loan Status Approved or not, namely Y- Yes, N-No | 
____
Importing Libraries and Datasets
First need to import the library:

- Pandas : For loading Dataframes
- Matplotlib : To visualize data features namely barplots
- Seaborn : To see the correlation between features using a heatmap
- Scikit-learn : library tools for predictive data analysis

____

### Analysis with Python

1. import the data "Loan Aproval.csv"
   
```html
<script>
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv('/Users/fredyfirmansyah/Downloads/Loan Aproval - Sheet1.csv')
data.head()

</script>
```
_
![image](https://fredyfirmansyah107.wordpress.com/wp-content/uploads/2024/08/screen-shot-2024-08-21-at-14.07.17.png?w=1024)

From the column above, we know that Loan_ID is unique and has no correlation with other column data, we can remove this from the dataset using .drop()
```html
<script>
# Loan_ID not have any corelation 
# Dropping Loan_ID coloumn

data.drop(['Loan_ID'],axis=1,inplace=True)

</script>
```
_
![image](https://fredyfirmansyah107.wordpress.com/wp-content/uploads/2024/08/screen-shot-2024-08-21-at-14.12.54.png?w=1024)


Now lets find out if there are any missing values ​​in the dataset using the code below.
```html
<script>
for col in data.columns :
    data[col] = data[col].fillna(data[col].mean())
    
data.isna().sum()
</script>

```
___
- Gender               0
- Married              0
- Dependents           0
- Education            0
- Self_Employed        0
- ApplicantIncome      0
- CoapplicantIncome    0
- LoanAmount           0
- Loan_Amount_Term     0
- Credit_History       0
- Property_Area        0
- Loan_Status          0
- dtype: int64
__



2. Visualize all the values ​​in a column using a barplot. It will simply show which values ​​dominate based on the dataset set.
```html
<script>
# show data with Barplot

obj = (data.dtypes == 'object')
object_cols = list(obj[obj].index)
plt.figure(figsize=(18,36))
index = 1

for col in object_cols:
    y = data[col].value_counts()
    plt.subplot(11,4,index)
    plt.xticks(rotation=90)
    sns.barplot(x=list(y.index), y=y)
    index +=1
    
</script>
```
_
![image](https://fredyfirmansyah107.wordpress.com/wp-content/uploads/2024/08/screen-shot-2024-08-21-at-14.42.52.png?w=1024)

Because all categorical values ​​are binary so we can use Label Encoder for all those columns and the values ​​will change to int data type.
* Check the object data type column. Let's find out if there is anything left.

```html
<script>

# Import Label Encoder
from sklearn import preprocessing

# label_encoder object knows how
# to understand word labels.

label_encoder = preprocessing.LabelEncoder()
obj = (data.dtypes == 'object')
for col in list(obj[obj].index):
    data[col] = label_encoder.fit_transform(data[col])

# To find the number of columns with
# datatype==object
obj = (data.dtypes == 'object')
print('Categorical variables:',len(list(obj[obj].index)))

</script>
```

Categorical variables: 0

Check the correlation between other column, heatmap is suitable to show the visual data

```html
<script>
plt.figure(figsize=(12,6))

sns.heatmap(data.corr(),cmap='BrBG',fmt='.2f',
            linewidths=2,annot=True)

</script>
```
_
![image](https://fredyfirmansyah107.wordpress.com/wp-content/uploads/2024/08/screen-shot-2024-08-21-at-14.58.01.png?w=1024)

Since there are no missing values ​​we should proceed to model training.
___

### Splitting Datasets
To map data, separate data is needed to make it easier so that the training and test process can get good results

 ```html
<script>
from sklearn.model_selection import train_test_split

X = data.drop(['Loan_Status'],axis=1)
Y = data['Loan_Status']
X.shape, Y.shape
         
X_train, X_test, Y_train, Y_test = train_test_split(X, Y,
                                                    test_size=0.4,
                                                    random_state=1)

X_train.shape, X_test.shape, Y_train.shape, Y_test.shape

</script>
```
_
![image](https://fredyfirmansyah107.wordpress.com/wp-content/uploads/2024/08/screen-shot-2024-08-21-at-15.28.09.png?w=344)


___
### Training and Evaluation Model

Since this is a classification problem we will use the following model:
- KNeighborsClassifiers
- RandomForestClassifiers
- Support Vector Classifiers (SVC)
- Logistics Regression

To predict accuracy we will use the accuracy score function from the scikit-learn library.
 ```html
<script>
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
log_model = LogisticRegression(solver='lbfgs', max_iter=1000)

from sklearn import metrics

knn = KNeighborsClassifier(n_neighbors=3)
rfc = RandomForestClassifier(n_estimators = 7,
                             criterion = 'entropy',
                             random_state = 7)

svc = SVC()
lc = LogisticRegression()

# Making Prediction on the training set
for clf in (rfc, knn, svc, lc):
    clf.fit(X_train, Y_train)
    Y_pred = clf.predict(X_train)
    print('accuracy score of ',
          clf.__class__.__name__,
          '=', 100*metrics.accuracy_score(Y_train,
                                          Y_pred))


</script>
```
_
![image](https://fredyfirmansyah107.wordpress.com/wp-content/uploads/2024/08/screen-shot-2024-08-21-at-15.39.27.png?w=1024)
 
### Results/Findings
The analysis results are summarized as follows:
Random Forest Classifier provided the best accuracy with an accuracy score more than 90% for the test dataset. And to get better results, ensemble learning techniques like Bagging and Boosting can also be used.

### Limitations









