# Iris Flower Classification
This is my machine learning project using Google Colab.
# Iris Flower Classification Project

# 1. Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 2. Load Dataset
iris = load_iris()

X = iris.data
y = iris.target

# Convert to DataFrame for better understanding
df = pd.DataFrame(X, columns=iris.feature_names)
df['target'] = y

print("First 5 rows:")
print(df.head())

# 3. Data Visualization
plt.scatter(df['sepal length (cm)'], df['sepal width (cm)'], c=y)
plt.xlabel("Sepal Length")
plt.ylabel("Sepal Width")
plt.title("Iris Dataset Visualization")
plt.show()

# 4. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 5. Model Training
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# 6. Predictions
y_pred = model.predict(X_test)

# 7. Evaluation
accuracy = accuracy_score(y_test, y_pred)
print("\nAccuracy:", accuracy)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# 8. Test with Custom Input
sample = [[5.1, 3.5, 1.4, 0.2]]  # Example flower
prediction = model.predict(sample)

print("\nCustom Prediction:", iris.target_names[prediction][0])
