import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import warnings
warnings.filterwarnings('ignore')

# (a) Load and handle missing values
df = pd.read_csv('DATA/heart_disease_dataset_01.csv')
df = df.fillna(df.median(numeric_only=True))
X = df.drop('target', axis=1)
y = df['target']

# (b) Correlation analysis
corr = df.corr()['target'].abs().sort_values(ascending=False)[1:6]
print("Top 5 Important Features:\n", corr)

# (c) Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# (d) Train models
rf = RandomForestClassifier(n_estimators=120, max_depth=8, random_state=42).fit(X_train, y_train)
svm = SVC(kernel='rbf', probability=True, random_state=42).fit(X_train, y_train)

# (e) Compute metrics
results = []
for name, model in [('Random Forest', rf), ('SVM', svm)]:
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    results.append({
        'Model': name,
        'Accuracy': round(accuracy_score(y_test, y_pred), 4),
        'Precision': round(precision_score(y_test, y_pred), 4),
        'Recall': round(recall_score(y_test, y_pred), 4),
        'F1-Score': round(f1_score(y_test, y_pred), 4),
        'AUC': round(roc_auc_score(y_test, y_pred_proba), 4)
    })

results_df = pd.DataFrame(results)
print("\n" + "="*70)
print(results_df.to_string(index=False))
print("="*70)

# (f) Plot accuracy comparison
plt.figure(figsize=(8, 5))
plt.bar(results_df['Model'], results_df['Accuracy'], color=['#3498db', '#e74c3c'], alpha=0.8)
plt.ylabel('Accuracy', fontsize=12)
plt.title('Model Accuracy Comparison', fontsize=14, fontweight='bold')
plt.ylim([0, 1])
for i, v in enumerate(results_df['Accuracy']):
    plt.text(i, v + 0.02, f'{v:.4f}', ha='center', fontsize=11)
plt.tight_layout()
plt.show()