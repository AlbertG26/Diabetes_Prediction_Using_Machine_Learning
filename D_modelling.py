import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# Load the dataset
file_path = 'path_to_your_dataset/cleaned_dataset.csv'
data = pd.read_csv(file_path)

# Step 1: Clean and Preprocess Data
# Drop rows with missing target values (if any)
data_cleaned = data.dropna(subset=['classification'])

# Clean the 'classification' column
data_cleaned['classification'] = data_cleaned['classification'].str.strip()  # Remove leading/trailing whitespace

# Map target variable to binary encoding
data_cleaned['classification'] = data_cleaned['classification'].map({'ckd': 1, 'notckd': 0})

# Separate features and target
X = data_cleaned.drop('classification', axis=1)
y = data_cleaned['classification']

# Handle missing values by imputing with the median for numerical columns
X = X.fillna(X.median())

# Encode categorical variables
X_encoded = pd.get_dummies(X, drop_first=True)

# Step 2: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42, stratify=y)

# Step 3: Model Training
rf_model = RandomForestClassifier(random_state=42, n_estimators=100)
rf_model.fit(X_train, y_train)

# Step 4: Predictions and Evaluation
y_pred = rf_model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
classification_report_summary = classification_report(y_test, y_pred)

# Print results
print("Accuracy:", accuracy)
print("\nClassification Report:\n", classification_report_summary)
