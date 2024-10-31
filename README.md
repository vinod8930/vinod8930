- import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_curve, auc
from sklearn.preprocessing import StandardScaler

# Load the Heart Disease dataset
data = pd.read_csv('C:/Users/vinod/Downloads/heart.csv')

# Display the first few rows and the column names
print(data.head())
print(data.columns)

# Define features and target variable
target_column = 'HeartDisease'  # Use 'HeartDisease' as the target column

# Remove the target column
X = data.drop(target_column, axis=1)  # Features
y = data[target_column]                # Target variable

# Convert categorical variables to dummy/indicator variables if needed
X = pd.get_dummies(X, drop_first=True)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Create an SVM classifier with a linear kernel
clf = SVC(kernel='rbf', random_state=42, probability=True)

# Fit the classifier to the training data
clf.fit(X_train, y_train)

# Make predictions on the test set
y_pred = clf.predict(X_test)

# Evaluate the classifier
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Calculate and print accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"\nAccuracy: {accuracy:.2f}")

# 1. Confusion Matrix Heatmap
confusion_mat = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(confusion_mat, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

# 2. ROC Curve
y_probs = clf.predict_proba(X_test)[:, 1]  # Probability estimates for the positive class
fpr, tpr, _ = roc_curve(y_test, y_probs)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', label=f'ROC Curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='red', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.show()

# 3. Bar Graph of Heart Disease Distribution
plt.figure(figsize=(6, 4))
data[target_column].value_counts().plot(kind='bar', color=['blue', 'orange'])
plt.title('Heart Disease Distribution')
plt.xlabel('Heart Disease')
plt.ylabel('Frequency')
plt.show()

# 4. Boolean Graph: Compare 'Cholesterol' above a threshold
threshold = 200
data['Cholesterol_above_threshold'] = data['Cholesterol'] > threshold
sns.countplot(data=data, x=target_column, hue='Cholesterol_above_threshold', palette=['blue', 'orange'])
plt.title(f'Cholesterol Above {threshold} by Heart Disease Outcome')
plt.xlabel('Heart Disease')
plt.ylabel('Count')
plt.legend(title=f'Cholesterol > {threshold}')
plt.show()

# 5. Histograms by Heart Disease Outcome
plt.figure(figsize=(24, 20))
for i, col in enumerate(data.columns[:-1], 1):
    plt.subplot(4, 4, i)
    sns.histplot(data=data, x=col, hue=target_column, multiple="stack", palette=['blue', 'orange'])
    plt.xlabel(col)
    plt.ylabel('Frequency')
plt.show()


# Normalize data for radar plot (after ensuring all features are present)
data = pd.get_dummies(data, drop_first=True)  # Apply get_dummies on the entire dataset
data_normalized = data.copy()

# Standardize features
data_normalized[X.columns] = scaler.fit_transform(data_normalized[X.columns])

# Define radar plot data
categories = X.columns
fig, ax = plt.subplots(1, 1, figsize=(8, 8), subplot_kw=dict(polar=True))

for outcome in data_normalized[target_column].unique():
    values = data_normalized[data_normalized[target_column] == outcome][categories].mean().tolist()
    values += values[:1]  # Close the radar chart
    ax.plot([n for n in range(len(categories))] + [0], values, label=f'Heart Disease {outcome}')
    ax.fill([n for n in range(len(categories))] + [0], values, alpha=0.2)

ax.set_xticks([n for n in range(len(categories))])
ax.set_xticklabels(categories)
plt.legend()
plt.title('Radar Plot of Feature Averages by Heart Disease Outcome')
plt.show()
