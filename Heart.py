
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler



df = pd.read_csv('DATA/heart_disease_dataset_01.csv')

num_features = df.shape[1] - 1
num_classes = df.iloc[:, -1].nunique()

print(f"Number of features: {num_features}")
print(f"Number of target classes: {num_classes}")
print(f"Target class distribution:\n{df.iloc[:, -1].value_counts()}\n")


features_to_plot = df.columns[:-1][:4].tolist()
target_col = df.columns[-1]


sns.set_style("whitegrid")
pairplot = sns.pairplot(df[features_to_plot + [target_col]],
                        hue=target_col,
                        palette="husl",
                        diag_kind="kde")
plt.suptitle('Pairwise Relationships (Seaborn)', y=1.001)
plt.tight_layout()
plt.savefig('pairplot_seaborn.png', dpi=100, bbox_inches='tight')
plt.show()



fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.ravel()

for idx, feature in enumerate(features_to_plot):
    colors = df[target_col].map({0: 'blue', 1: 'red'})
    axes[idx].scatter(df[feature], df.iloc[:, 1], c=colors, alpha=0.6, edgecolors='k')
    axes[idx].set_xlabel(feature)
    axes[idx].set_ylabel(df.columns[1])
    axes[idx].set_title(f'{feature} vs {df.columns[1]}')
    axes[idx].grid(True, alpha=0.3)

handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=8, label='No Disease'),
           plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=8, label='Disease')]
fig.legend(handles=handles, loc='upper right', bbox_to_anchor=(1.15, 0.95))
plt.suptitle('Pairwise Relationships (Matplotlib)', fontsize=14)
plt.tight_layout()
plt.savefig('pairplot_matplotlib.png', dpi=100, bbox_inches='tight')
plt.show()


missing_values = df.isnull().sum()
print(f"Missing values per column:\n{missing_values}\n")

if missing_values.sum() > 0:
    print("→ Replacing missing values with median...")
    for column in df.columns[:-1]:  # Exclude target
        if df[column].isnull().sum() > 0:
            median_value = df[column].median()
            df[column].fillna(median_value, inplace=True)
            print(f"  Column '{column}': filled with median {median_value}")
    print("✓ Missing values replaced\n")
else:
    print("✓ No missing values found in dataset\n")


X = df.iloc[:, :-1]
y = df.iloc[:, -1]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

print(f"Original feature mean (first 3): {X.iloc[:, :3].mean().values}")
print(f"Scaled feature mean (first 3): {X_scaled.iloc[:, :3].mean().values}")
print(f"Original feature std (first 3): {X.iloc[:, :3].std().values}")
print(f"Scaled feature std (first 3): {X_scaled.iloc[:, :3].std().values}")
print("✓ Features standardized using StandardScaler\n")



# Split 1: 70-30
X_train_70, X_test_30, y_train_70, y_test_30 = train_test_split(
    X_scaled, y, test_size=0.30, random_state=42, stratify=y
)
print(f"Split 1 (70-30):")
print(f"  Train set: {X_train_70.shape[0]} samples")
print(f"  Test set: {X_test_30.shape[0]} samples\n")

# Split 2: 80-20
X_train_80, X_test_20, y_train_80, y_test_20 = train_test_split(
    X_scaled, y, test_size=0.20, random_state=42, stratify=y
)
print(f"Split 2 (80-20):")
print(f"  Train set: {X_train_80.shape[0]} samples")
print(f"  Test set: {X_test_20.shape[0]} samples")
print("✓ Dataset split completed\n")


lr_model_70 = LogisticRegression(max_iter=1000, random_state=42)
lr_model_70.fit(X_train_70, y_train_70)
y_pred_70 = lr_model_70.predict(X_test_30)
print(f"Model coefficients shape: {lr_model_70.coef_.shape}\n")



X_original = df.iloc[:, :-1]
y_original = df.iloc[:, -1]

plt.figure(figsize=(12, 7))
colors = y_original.map({0: 'blue', 1: 'red'})
scatter = plt.scatter(X_original['age'], X_original['cholesterol'],
                      c=colors, alpha=0.6, s=100, edgecolors='black', linewidth=0.5)

plt.xlabel('Age', fontsize=12)
plt.ylabel('Chol', fontsize=12)
plt.title('Age vs Cholesterol (Heart Disease Status)', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)

handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='blue',
                      markersize=10, label='No Disease (0)'),
           plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red',
                      markersize=10, label='Heart Disease (1)')]
plt.legend(handles=handles, fontsize=11, loc='upper right')

plt.tight_layout()
plt.savefig('age_vs_cholesterol.png', dpi=100, bbox_inches='tight')
plt.show()




accuracy_70 = accuracy_score(y_test_30, y_pred_70)
precision_70 = precision_score(y_test_30, y_pred_70)
recall_70 = recall_score(y_test_30, y_pred_70)
f1_70 = f1_score(y_test_30, y_pred_70)
auc_70 = roc_auc_score(y_test_30, lr_model_70.predict_proba(X_test_30)[:, 1])

print(f"Accuracy:  {accuracy_70:.4f}")
print(f"Precision: {precision_70:.4f}")
print(f"Recall:    {recall_70:.4f}")
print(f"F1-Score:  {f1_70:.4f}")
print(f"AUC Score: {auc_70:.4f}\n")



lr_model_80 = LogisticRegression(max_iter=1000, random_state=42)
lr_model_80.fit(X_train_80, y_train_80)
y_pred_80 = lr_model_80.predict(X_test_20)

accuracy_80 = accuracy_score(y_test_20, y_pred_80)
precision_80 = precision_score(y_test_20, y_pred_80)
recall_80 = recall_score(y_test_20, y_pred_80)
f1_80 = f1_score(y_test_20, y_pred_80)
auc_80 = roc_auc_score(y_test_20, lr_model_80.predict_proba(X_test_20)[:, 1])

print(f"Accuracy:  {accuracy_80:.4f}")
print(f"Precision: {precision_80:.4f}")
print(f"Recall:    {recall_80:.4f}")
print(f"F1-Score:  {f1_80:.4f}")
print(f"AUC Score: {auc_80:.4f}\n")



comparison_df = pd.DataFrame({
    '70-30 Split': [accuracy_70, precision_70, recall_70, f1_70, auc_70],
    '80-20 Split': [accuracy_80, precision_80, recall_80, f1_80, auc_80]
}, index=['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC'])

print(comparison_df)
print()


fig, axes = plt.subplots(1, 2, figsize=(12, 4))


cm_70 = confusion_matrix(y_test_30, y_pred_70)
sns.heatmap(cm_70, annot=True, fmt='d', cmap='Blues', ax=axes[0], cbar=False)
axes[0].set_title('Confusion Matrix (70-30 Split)', fontweight='bold')
axes[0].set_ylabel('True Label')
axes[0].set_xlabel('Predicted Label')


cm_80 = confusion_matrix(y_test_20, y_pred_80)
sns.heatmap(cm_80, annot=True, fmt='d', cmap='Blues', ax=axes[1], cbar=False)
axes[1].set_title('Confusion Matrix (80-20 Split)', fontweight='bold')
axes[1].set_ylabel('True Label')
axes[1].set_xlabel('Predicted Label')

plt.tight_layout()
plt.savefig('confusion_matrices.png', dpi=100, bbox_inches='tight')
plt.show()
print("✓ Confusion matrices saved as 'confusion_matrices.png'\n")
