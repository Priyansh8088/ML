import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler

# 1. Load Titanic dataset
df = pd.read_csv('DATA/titanic.csv')

# Select features
features = ['Pclass', 'Sex', 'Age', 'Fare', 'Embarked']
target = 'Survived'
df = df[features + [target]].dropna(subset=[target])

X = df[features]
y = df[target]

# Split 80-20
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. Data preparation
# Numerical & categorical features
num_features = ['Age', 'Fare']
cat_features = ['Pclass', 'Sex', 'Embarked']

# Impute Age with median
imputer = SimpleImputer(strategy='median')
X_train[num_features] = imputer.fit_transform(X_train[num_features])
X_test[num_features] = imputer.transform(X_test[num_features])

# One-hot encode categorical features
encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
X_train_cat = encoder.fit_transform(X_train[cat_features])
X_test_cat = encoder.transform(X_test[cat_features])

# MinMaxScale numerical features
scaler = MinMaxScaler()
X_train_num = scaler.fit_transform(X_train[num_features])
X_test_num = scaler.transform(X_test[num_features])

# Combine
X_train_proc = np.hstack([X_train_cat, X_train_num])
X_test_proc = np.hstack([X_test_cat, X_test_num])

# 3. Train Naive Bayes classifier
nb = GaussianNB()
nb.fit(X_train_proc, y_train)

# Predict
y_pred = nb.predict(X_test_proc)
y_pred_proba = nb.predict_proba(X_test_proc)[:, 1]

# Display first 10 predictions
print("First 10 Actual vs Predicted:")
print(pd.DataFrame({'Actual': y_test.values[:10], 'Predicted': y_pred[:10]}))

# 4. Classification metrics
print("\n--- Classification Metrics ---")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision (macro): {precision_score(y_test, y_pred, average='macro'):.4f}")
print(f"Recall (macro): {recall_score(y_test, y_pred, average='macro'):.4f}")
print(f"F1-Score (macro): {f1_score(y_test, y_pred, average='macro'):.4f}")

# 5. Visualizations
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Survived vs Not-Survived count plot
ax1 = axes[0, 0]
survive_counts = pd.Series(y_test).value_counts()
ax1.bar(['Not Survived', 'Survived'], [survive_counts[0], survive_counts[1]], color=['red', 'green'])
ax1.set_title('Survived vs Not-Survived Count Plot')
ax1.set_ylabel('Count')

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
ax2 = axes[0, 1]
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax2, cbar=False)
ax2.set_title('Confusion Matrix')
ax2.set_ylabel('Actual')
ax2.set_xlabel('Predicted')

# ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
roc_auc = auc(fpr, tpr)
ax3 = axes[1, 0]
ax3.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
ax3.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
ax3.set_xlim([0.0, 1.0])
ax3.set_ylim([0.0, 1.05])
ax3.set_xlabel('False Positive Rate')
ax3.set_ylabel('True Positive Rate')
ax3.set_title('ROC Curve')
ax3.legend(loc="lower right")

# Hide unused subplot
axes[1, 1].axis('off')

plt.tight_layout()
plt.show()

print(f"\nAUC: {roc_auc:.4f}")