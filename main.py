import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("Diabetes.csv")

print("First 5 rows of the dataset:")
print(df.head())

print("\nDataset Information:")
print(df.info())

print("\nStatistical Summary of the dataset:")
print(df.describe())

columns_to_check = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
print("\nCount of zeros in columns where they are likely invalid:")
print((df[columns_to_check] == 0).sum())

df[columns_to_check] = df[columns_to_check].replace(0, np.nan)

imputer = SimpleImputer(strategy='median')
df[columns_to_check] = pd.DataFrame(
    imputer.fit_transform(df[columns_to_check]),
    columns=columns_to_check,
    index=df.index
)

print("\nData after replacing zeros with median:")
print(df[columns_to_check].head())

for col in columns_to_check:
    plt.figure(figsize=(8, 4))
    sns.boxplot(x=df[col])
    plt.title(f"Boxplot of {col} (Outlier Detection)")
    plt.show()

for col in columns_to_check:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]

print("\nShape of the dataset after outlier removal:")
print(df.shape)

df.to_csv("Diabetes_cleaned.csv", index=False)
print("\nCleaned dataset saved as 'diabetes_cleaned.csv'")

for col in columns_to_check:
    plt.figure(figsize=(8, 4))
    sns.histplot(df[col], kde=True, bins=20)
    plt.title(f"Distribution of {col} (After Cleaning)")
    plt.show()

print("\nCheck for any remaining missing values:")
print(df.isna().sum())
