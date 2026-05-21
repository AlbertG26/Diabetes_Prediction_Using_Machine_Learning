### Diabetes Prediction Using Machine Learning
A data cleaning, exploration, and predictive modelling project

### Overview
This project explores an open‑source diabetes dataset and demonstrates how data preprocessing, exploratory data analysis (EDA), and machine learning can be used to predict diabetes outcomes. The work focuses on improving data quality, identifying key predictive features, and evaluating two classification models: Logistic Regression and Random Forest.

The project was completed as part of the Data Science COM618 module and highlights practical skills in Python, data cleaning, visualisation, and model evaluation.

### Project Objectives
Clean and prepare an open‑source diabetes dataset

Explore patterns, distributions, and correlations using visualisation

Build predictive models to classify diabetic vs non‑diabetic patients

Evaluate model performance using confusion matrices and ROC curves

Discuss limitations, challenges, and opportunities for improvement

### Dataset
The dataset contains 768 patient records with 9 clinical and demographic features, plus a binary outcome:

Independent variables: Pregnancies, Glucose, Blood Pressure, Skin Thickness, Insulin, BMI, Diabetes Pedigree Function, Age

Target variable: Outcome (1 = diabetic, 0 = non‑diabetic)

The dataset is publicly available on Kaggle and widely used in diabetes prediction research.

### Methods & Preprocessing
The dataset required several cleaning steps to ensure reliable modelling:

### Handling Missing & Invalid Values
Columns such as Glucose, Blood Pressure, Skin Thickness, Insulin, and BMI contained zeros, which are not clinically valid.
These were replaced using median imputation to reduce the influence of outliers.

### Outlier Detection & Removal
Outliers were identified using the Interquartile Range (IQR) method and removed to improve model stability.

### Feature Scaling
Numeric features were normalised to ensure consistent scaling for algorithms sensitive to magnitude differences.

### Exploratory Data Analysis (EDA)
Key insights from the EDA:

Glucose showed the strongest correlation with diabetes outcome

BMI and Age had moderate positive correlations

Several features displayed skewed distributions

The dataset was slightly imbalanced (65% non‑diabetic, 35% diabetic)

Visualisations included histograms, boxplots, correlation heatmaps, and class distribution charts.

### Machine Learning Models
Two models were implemented:

1. Logistic Regression
Used as a baseline model

Simple, interpretable, and effective for binary classification

ROC curve and AUC used for evaluation

2. Random Forest Classifier
Ensemble method capturing non‑linear relationships

Improved accuracy and robustness

Provided feature importance insights (Glucose and BMI ranked highest)

### Evaluation
Model performance was assessed using:

Confusion Matrix

Accuracy Score

ROC Curve (Logistic Regression)

Feature Importance (Random Forest)

Random Forest outperformed Logistic Regression, particularly in handling feature interactions and non‑linear patterns.

### Limitations
Zero‑value imputation may not fully capture clinical reality

Outlier removal risks discarding valid extreme cases

Dataset slightly imbalanced

No external validation dataset used

More advanced models (e.g., XGBoost, SVM) were not explored

No feature engineering beyond cleaning and scaling

### Future Improvements
Potential enhancements include:

Using KNN imputation or MICE for more realistic missing value handling

Applying SMOTE to address class imbalance

Testing advanced models like XGBoost or Gradient Boosting

Adding domain‑informed features (e.g., glucose/BMI thresholds)

Validating the model on external datasets

### Technologies Used
Python

Pandas, NumPy

Scikit‑learn

Seaborn, Matplotlib

Jupyter Notebook

### How to Run
Clone the repository

Install dependencies:

bash
pip install -r requirements.txt
Open the notebook:

bash
jupyter notebook
Run the cells in order to reproduce the analysis
