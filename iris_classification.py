# Iris Flower Classification
# This script uses the Iris dataset to train a machine learning model
# to classify iris flowers into their respective species

# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

# Load the Iris dataset from scikit-learn
iris = datasets.load_iris()
X = iris.data  # Features: sepal length, sepal width, petal length, petal width
y = iris.target  # Target: species (0: setosa, 1: versicolor, 2: virginica)

# Create a DataFrame for better data manipulation
iris_df = pd.DataFrame(X, columns=iris.feature_names)
iris_df['species'] = pd.Categorical.from_codes(y, iris.target_names)

# Display basic information about the dataset
print("\nIris Dataset Information:")
print(f"Number of samples: {len(iris_df)}")
print(f"Number of features: {len(iris.feature_names)}")
print(f"Number of classes: {len(iris.target_names)}")
print(f"Class distribution:\n{iris_df['species'].value_counts()}")

# Display the first few rows of the dataset
print("\nFirst 5 rows of the dataset:")
print(iris_df.head())

# Split the data into features and target
X = iris_df.drop('species', axis=1)
y = iris_df['species'].cat.codes

# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train a Random Forest classifier
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train_scaled, y_train)

# Make predictions
y_pred = rf_model.predict(X_test_scaled)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred, target_names=iris.target_names)

print("\nRandom Forest Model Evaluation:")
print(f"Accuracy: {accuracy:.4f}")
print("\nConfusion Matrix:")
print(conf_matrix)
print("\nClassification Report:")
print(class_report)

# Compare with other classifiers
classifiers = {
    'Support Vector Machine': SVC(random_state=42),
    'K-Nearest Neighbors': KNeighborsClassifier(),
    'Decision Tree': DecisionTreeClassifier(random_state=42)
}

print("\nComparing different classifiers:")
for name, clf in classifiers.items():
    clf.fit(X_train_scaled, y_train)
    y_pred = clf.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"{name} Accuracy: {accuracy:.4f}")

# Visualize the data
plt.figure(figsize=(12, 5))

# Pairplot to visualize relationships between features
plt.subplot(1, 2, 1)
sns.scatterplot(x='sepal length (cm)', y='sepal width (cm)', 
                hue='species', data=iris_df, palette='viridis')
plt.title('Sepal Length vs Sepal Width')

plt.subplot(1, 2, 2)
sns.scatterplot(x='petal length (cm)', y='petal width (cm)', 
                hue='species', data=iris_df, palette='viridis')
plt.title('Petal Length vs Petal Width')

plt.tight_layout()
plt.savefig('iris_visualization.png')

# Feature importance from Random Forest
feature_importance = pd.DataFrame({
    'Feature': iris.feature_names,
    'Importance': rf_model.feature_importances_
})
feature_importance = feature_importance.sort_values('Importance', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importance, palette='viridis')
plt.title('Feature Importance from Random Forest')
plt.tight_layout()
plt.savefig('feature_importance.png')

print("\nVisualization images saved as 'iris_visualization.png' and 'feature_importance.png'")