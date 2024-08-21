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







  

