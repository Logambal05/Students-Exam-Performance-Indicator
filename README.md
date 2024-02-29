# Student Exam Performance Indicator

## Problem Statement
This project aims to understand how student performance (test scores) is affected by various variables such as Gender, Ethnicity, Parental Level of Education, Lunch, and Test Preparation Course.

## Dataset Information
The dataset used in this project includes the following features:
- Gender: Sex of students (Male/Female).
- Race/Ethnicity: Ethnicity of students (Group A, B, C, D, E).
- Parental Level of Education: Parents' final education (Bachelor's Degree, Some College, Master's Degree, Associate's Degree, High School).
- Lunch: Whether the student had lunch before the test (Standard or Free/Reduced).
- Test Preparation Course: Whether the student completed a test preparation course before the test.
- Math Score: Score of a particular student in math.
- Reading Score: Score of a particular student in reading.
- Writing Score: Score of a particular student in writing.

## Project Overview
This project consists of the following steps:

1. **Data Cleaning**: The dataset was cleaned to handle missing values, outliers, and inconsistencies. This involved imputation, outlier removal, and data normalization.
   
2. **Exploratory Data Analysis (EDA)**: Exploratory data analysis was performed to gain insights into the relationships between different variables and identify patterns in the data.

3. **Feature Engineering**: Additional features were created or transformed to enhance the predictive power of the model. This may include feature scaling, one-hot encoding, or feature extraction.

4. **Model Training**: Several machine learning models were trained and evaluated to predict student performance based on the given features. Models such as linear regression, decision trees, and ensemble methods were considered.

5. **Model Evaluation**: The performance of each model was evaluated using appropriate evaluation metrics such as Mean Absolute Error (MAE), Mean Squared Error (MSE), and R-squared. Cross-validation techniques were employed to ensure robustness of the models.

6. **Model Deployment**: The best performing model was deployed using Flask on AWS Beanstalk to create a web application for predicting student performance.

7. **Web Application**: A user-friendly web interface was developed where users can input student information and get predictions for student performance.

## Tools Used
- Python: For data cleaning, analysis, and model training.
- Pandas: For data manipulation and cleaning.
- Matplotlib and Seaborn: For data visualization.
- Scikit-learn: For machine learning model training and evaluation.
- Flask: For web application development.
- AWS Beanstalk: For model deployment.

## AWS Deployment Link
Aws Beanstalk Link : (http://studentsexamperformaceindicator-env.eba-eainjvrs.us-east-1.elasticbeanstalk.com/predictdata)


## Screenshots of UI
UI Image: ![UI Image](https://drive.google.com/uc?export=view&id=1Q8vn-eZ9wx8bQdbFkz5TEIoTfzXiALeS)
Deployment Image: ![Deployment Image](https://drive.google.com/uc?export=view&id=1hwZ3KJcVQaAmjBYwY8t8z9Ni8KZ77WwP)

## Conclusion
This project successfully developed a predictive model for student performance based on various demographic and educational factors. The deployed web application provides a user-friendly interface for stakeholders such as educators and policymakers to gain insights into student performance and make informed decisions.

