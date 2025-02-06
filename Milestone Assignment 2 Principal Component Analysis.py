# Step 1: Install Necessary Libraries
import subprocess
import sys

# Function to install packages
def install(package):
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])

# List of required packages
required_packages = ['numpy', 'pandas', 'matplotlib', 'seaborn', 'scikit-learn']

# Install each package if not already installed
for package in required_packages:
    try:
        __import__(package)
    except ImportError:
        print(f"Installing {package}...")
        install(package)

# Step 2: Import Necessary Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# ----------------------------------------------
# Milestone Assignment 2: Principal Component Analysis (PCA)
#  Task:
# 1️ Load Breast Cancer dataset
# 2️ Perform PCA to reduce to 2 principal components
# 3️ Visualize PCA components
# 4️ (Bonus) Train Logistic Regression on PCA data
# ----------------------------------------------

# Step 3: Load and Explore the Dataset
cancer = load_breast_cancer()  # Load dataset

# Convert to Pandas DataFrame
df = pd.DataFrame(cancer.data, columns=cancer.feature_names)
df['target'] = cancer.target  # Add target column (0 = Malignant, 1 = Benign)

# Display dataset info
print("Dataset Shape:", df.shape)
print("Feature Names:", df.columns[:-1])
print("Class Distribution:\n", df['target'].value_counts())

#  Step 4: Standardize the Data
# PCA requires data to be normalized (mean=0, variance=1)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df.iloc[:, :-1])  # Exclude target column

#  Step 5: Apply PCA (Reduce to 2 Components)
pca = PCA(n_components=2)  # Keep 2 components
X_pca = pca.fit_transform(X_scaled)  # Transform dataset

# Convert PCA result into DataFrame
pca_df = pd.DataFrame(X_pca, columns=['PC1', 'PC2'])
pca_df['target'] = df['target']  # Add target column

#  Step 6: Visualize PCA Components
plt.figure(figsize=(10, 6))
sns.scatterplot(x=pca_df['PC1'], y=pca_df['PC2'], hue=pca_df['target'], palette='coolwarm')
plt.title('PCA of Breast Cancer Dataset')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend(title='Target', loc='best')
plt.show()

# 📌 Step 7 (Bonus): Train Logistic Regression on PCA Data
# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_pca, df['target'], test_size=0.2, random_state=42)

# Train Logistic Regression model
log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)

# Predict on test data
y_pred = log_reg.predict(X_test)

# Evaluate model performance
accuracy = accuracy_score(y_test, y_pred)
print(f"Logistic Regression Accuracy on PCA Data: {accuracy:.2f}")

#  Step 8: Save Transformed Data
pca_df.to_csv("pca_transformed_data.csv", index=False)
print("PCA Transformed Data saved as 'pca_transformed_data.csv'")
