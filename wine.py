import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

# Load dataset
df = pd.read_csv('DATA/wine_dataset.csv')

# Split data (80-20)
X = df.drop('target', axis=1)
y = df['target']
X_train_df, X_test_df, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# a) Print sizes
print(f"Training set size: {len(X_train_df)}")
print(f"Testing set size: {len(X_test_df)}\n")

# b) Correlation heatmap (use DataFrame to access .corr)
plt.figure(figsize=(10, 8))
sns.heatmap(X_train_df.corr(), cmap='coolwarm', center=0, square=True)
plt.title('Correlation Matrix - Wine Dataset')
plt.tight_layout()
plt.show()

# Scale features (keep DataFrame with column names after scaling)
scaler = StandardScaler()
X_train = pd.DataFrame(
    scaler.fit_transform(X_train_df),
    columns=X_train_df.columns,
    index=X_train_df.index
)
X_test = pd.DataFrame(
    scaler.transform(X_test_df),
    columns=X_test_df.columns,
    index=X_test_df.index
)

# c) & d) Train models and collect metrics
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42, multi_class='ovr'),
    'Decision Tree': DecisionTreeClassifier(max_depth=4, random_state=42)
}

results = []

for model_name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    # predict_proba returns (n_samples, n_classes), required for multiclass AUC
    y_pred_proba = model.predict_proba(X_test)

    # Calculate metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr')

    results.append({
        'Model': model_name,
        'Accuracy': f'{acc:.4f}',
        'Precision': f'{prec:.4f}',
        'Recall': f'{rec:.4f}',
        'F1-Score': f'{f1:.4f}',
        'AUC': f'{auc:.4f}'
    })

# d) Display results table
results_df = pd.DataFrame(results)
print(results_df.to_string(index=False))
print()

# e) Plot accuracy comparison
accuracies = [float(r['Accuracy']) for r in results]
plt.figure(figsize=(8, 5))
plt.bar(results_df['Model'], accuracies, color=['#1f77b4', '#ff7f0e'])
plt.ylabel('Accuracy', fontsize=12)
plt.title('Accuracy Comparison: Logistic Regression vs Decision Tree', fontsize=14)
plt.ylim([0, 1])
for i, v in enumerate(accuracies):
    plt.text(i, v + 0.02, f'{v:.4f}', ha='center', fontsize=11)
plt.tight_layout()
plt.show()