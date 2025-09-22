# Iris Flower Classification

## Overview
This project implements a machine learning model to classify Iris flowers into their respective species (setosa, versicolor, virginica) based on their measurements. The Iris dataset is a classic dataset in the field of machine learning and pattern recognition.

## Dataset
The Iris dataset contains measurements for 150 iris flowers from three different species:
- Iris Setosa
- Iris Versicolor
- Iris Virginica

For each flower, four features are measured:
- Sepal length (cm)
- Sepal width (cm)
- Petal length (cm)
- Petal width (cm)

## Project Structure
- `iris_classification.py`: Main script that loads the data, trains models, and evaluates performance
- `iris_visualization.png`: Visualization of the dataset features
- `feature_importance.png`: Visualization of feature importance from the Random Forest model

## Implementation Details

### Data Preprocessing
- The dataset is loaded using scikit-learn's built-in datasets module
- Data is split into training (80%) and testing (20%) sets
- Features are standardized using StandardScaler

### Models Implemented
- Random Forest Classifier (primary model)
- Support Vector Machine
- K-Nearest Neighbors
- Decision Tree

### Evaluation Metrics
- Accuracy
- Confusion Matrix
- Classification Report (Precision, Recall, F1-score)

## Results
The Random Forest classifier achieves high accuracy in classifying the Iris flowers. The model performance metrics and visualizations are generated when running the script.

## How to Run

```bash
python iris_classification.py
```

This will:
1. Load and preprocess the Iris dataset
2. Train multiple classification models
3. Evaluate and compare model performance
4. Generate visualizations of the data and feature importance

## Requirements
- Python 3.x
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn

You can install the required packages using:
```bash
pip install numpy pandas matplotlib seaborn scikit-learn
```