import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("Diabetes_cleaned.csv")

print("First 5 rows of the dataset:")
print(df.head())

print("\nSummary Statistics:")
print(df.describe())

print("\nDistribution of Outcome (0: No Diabetes, 1: Diabetes):")
print(df['Outcome'].value_counts())

plt.figure(figsize=(10, 8))
correlation_matrix = df.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Matrix")
plt.show()

numerical_columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']

for col in numerical_columns:
    plt.figure(figsize=(8, 4))
    sns.histplot(df[col], kde=True, bins=20)
    plt.title(f"Distribution of {col}")
    plt.show()

for col in numerical_columns:
    plt.figure(figsize=(8, 4))
    sns.boxplot(x='Outcome', y=col, data=df)
    plt.title(f"{col} vs Outcome")
    plt.show()


plt.figure(figsize=(6, 4))
sns.countplot(x='Outcome', data=df)
plt.title("Distribution of Outcome (0: No Diabetes, 1: Diabetes)")
plt.xlabel("Outcome")
plt.ylabel("Count")
plt.show()
