import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# Load data
data = load_breast_cancer()
X, y = pd.DataFrame(data.data, columns=data.feature_names), data.target

# (a) Dataset overview
print(f"Dataset Shape: {X.shape}")
print(f"Features: {list(X.columns)}\n")
print(f"Class Distribution:\n{pd.Series(y, name='Target').value_counts()}\n")

# Correlation heatmap
corr = pd.concat([X, pd.Series(y, name='Target', index=X.index)], axis=1).corr()
top5 = corr['Target'].abs().nlargest(6)[1:].index.tolist()
print(f"Top 5 Correlated Features: {top5}\n")

plt.figure(figsize=(10, 8))
sns.heatmap(X[top5].corr(), annot=True, fmt='.2f', cmap='coolwarm', cbar_kws={'label': 'Correlation'})
plt.title('Correlation Heatmap - Top 5 Features')
plt.tight_layout()
plt.show()

# (b) Missing values & preprocessing
X_missing = X.copy()
mask = np.random.rand(*X_missing.shape) < 0.03
X_missing[mask] = np.nan
print(f"Missing values introduced: {X_missing.isna().sum().sum()}")

imputer = SimpleImputer(strategy='median')
X_imputed = pd.DataFrame(imputer.fit_transform(X_missing), columns=X.columns)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imputed)

# (c) Two data splits & visualization
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# 70-30 split
X_train1, X_test1, y_train1, y_test1 = train_test_split(X_scaled, y, test_size=0.3, random_state=42)
# 80-20 split
X_train2, X_test2, y_train2, y_test2 = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Visualizations
col_mean_radius = list(X.columns).index('mean radius')
col_mean_texture = list(X.columns).index('mean texture')
col_mean_area = list(X.columns).index('mean area')

# Scatter plot (70-30)
axes[0, 0].scatter(X_train1[:, col_mean_radius], X_train1[:, col_mean_texture], c=y_train1, cmap='RdYlGn_r', alpha=0.7)
axes[0, 0].set_xlabel('Mean Radius'), axes[0, 0].set_ylabel('Mean Texture')
axes[0, 0].set_title('70-30 Split: Mean Radius vs Mean Texture')

# Scatter plot (80-20)
axes[0, 1].scatter(X_train2[:, col_mean_radius], X_train2[:, col_mean_texture], c=y_train2, cmap='RdYlGn_r', alpha=0.7)
axes[0, 1].set_xlabel('Mean Radius'), axes[0, 1].set_ylabel('Mean Texture')
axes[0, 1].set_title('80-20 Split: Mean Radius vs Mean Texture')

# Box plots
axes[1, 0].boxplot([X_train1[y_train1 == 0, col_mean_area], X_train1[y_train1 == 1, col_mean_area]],
                   tick_labels=['Benign', 'Malignant'])
axes[1, 0].set_ylabel('Mean Area'), axes[1, 0].set_title('70-30 Split: Mean Area Distribution')

axes[1, 1].boxplot([X_train2[y_train2 == 0, col_mean_area], X_train2[y_train2 == 1, col_mean_area]],
                   tick_labels=['Benign', 'Malignant'])
axes[1, 1].set_ylabel('Mean Area'), axes[1, 1].set_title('80-20 Split: Mean Area Distribution')

plt.tight_layout()
plt.show()

# (d) Train and evaluate both models on both splits
results = []
for split_name, (X_tr, X_te, y_tr, y_te) in [('70-30', (X_train1, X_test1, y_train1, y_test1)),
                                             ('80-20', (X_train2, X_test2, y_train2, y_test2))]:
    for model_name, model in [('Logistic Regression', LogisticRegression(max_iter=1000)),
                              ('Decision Tree', DecisionTreeClassifier(random_state=42))]:
        model.fit(X_tr, y_tr)
        y_pred = model.predict(X_te)
        y_proba = model.predict_proba(X_te)[:, 1]

        results.append({
            'Split': split_name,
            'Model': model_name,
            'Accuracy': accuracy_score(y_te, y_pred),
            'Precision': precision_score(y_te, y_pred),
            'Recall': recall_score(y_te, y_pred),
            'F1-Score': f1_score(y_te, y_pred),
            'ROC-AUC': roc_auc_score(y_te, y_proba)
        })

results_df = pd.DataFrame(results)
print(results_df.to_string(index=False))
