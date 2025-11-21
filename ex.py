
# =================================================================
# 1. CALIFORNIA HOUSING (REGRESSION)
# =================================================================

print("=" * 70)
print("1. CALIFORNIA HOUSING - REGRESSION PIPELINE")
print("=" * 70)

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 1.1 Load data
data = fetch_california_housing(as_frame=True)
df = data.frame
print(f"\n1.1 Dataset shape: {df.shape}")

# 1.2 Train-test split
X = df.drop('MedHouseVal', axis=1)
y = df['MedHouseVal']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"1.2 Train: {X_train.shape}, Test: {X_test.shape}")

# 1.3 Median imputation
imputer = SimpleImputer(strategy='median')
X_train_imputed = imputer.fit_transform(X_train)
X_test_imputed = imputer.transform(X_test)
print(f"1.3 Missing values handled")

# 1.4 Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_imputed)
X_test_scaled = scaler.transform(X_test_imputed)
print(f"1.4 Features scaled")

# 1.5 Train Decision Tree
model = DecisionTreeRegressor(max_depth=5, random_state=42)
model.fit(X_train_scaled, y_train)
print(f"1.5 Model trained")

# 1.6 Predictions
y_pred = model.predict(X_test_scaled)
print(f"\n1.6 Sample predictions:")
print(pd.DataFrame({'Actual': y_test[:5].values, 'Predicted': y_pred[:5]}))

# 1.7 Evaluation metrics
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)
print(f"\n1.7 Metrics: MAE={mae:.4f}, RMSE={rmse:.4f}, R²={r2:.4f}")

# 1.8 Actual vs Predicted plot
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('CalHouse: Actual vs Predicted')
plt.tight_layout()
plt.savefig('calhouse_predictions.png')
print("1.8 Prediction plot saved")

# 1.9 Feature importance
importances = model.feature_importances_
feature_names = X.columns
top5_idx = np.argsort(importances)[-5:]
plt.figure(figsize=(8, 5))
plt.barh(feature_names[top5_idx], importances[top5_idx])
plt.xlabel('Importance')
plt.title('Top 5 Important Features')
plt.tight_layout()
plt.savefig('calhouse_importance.png')
print("1.9 Feature importance plot saved\n")


# =================================================================
# 2. BREAST CANCER (BINARY CLASSIFICATION)
# =================================================================

print("=" * 70)
print("2. BREAST CANCER - BINARY CLASSIFICATION")
print("=" * 70)

from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import seaborn as sns

# 2.1 Load dataset
data = load_breast_cancer()
X = data.data
y = data.target
print(f"\n2.1 Shape: {X.shape}, Classes: {np.unique(y)}")

# 2.2 Dataset overview
print(f"2.2 Class distribution: {np.bincount(y)}")
df_cancer = pd.DataFrame(X, columns=data.feature_names)
df_cancer['target'] = y
correlations = df_cancer.corr()['target'].sort_values(ascending=False)
print(f"Top 5 correlated features:\n{correlations[1:6]}")

# 2.3 Correlation heatmap
top5_features = correlations[1:6].index.tolist()
plt.figure(figsize=(8, 6))
sns.heatmap(df_cancer[top5_features + ['target']].corr(), annot=True, fmt='.2f', cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.tight_layout()
plt.savefig('cancer_correlation.png')
print("2.3 Correlation heatmap saved")

# 2.4 Add missing values (simulate ~3%)
X_missing = X.copy()
mask = np.random.rand(*X.shape) < 0.03
X_missing[mask] = np.nan
imputer = SimpleImputer(strategy='median')
X_imputed = imputer.fit_transform(X_missing)
print("2.4 Missing values simulated and imputed")

# 2.5 Standardization
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imputed)
print("2.5 Features standardized")

# 2.6 Two data splits
X_train1, X_test1, y_train1, y_test1 = train_test_split(X_scaled, y, test_size=0.3, random_state=42, stratify=y)
X_train2, X_test2, y_train2, y_test2 = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)
print(f"2.6 Split1: {X_train1.shape}, Split2: {X_train2.shape}")

# 2.7 Visualizations - Scatter plots
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].scatter(X_train1[:, 0], X_train1[:, 1], c=y_train1, cmap='viridis', alpha=0.6)
axes[0].set_title('Split 1 (70/30)')
axes[1].scatter(X_train2[:, 0], X_train2[:, 1], c=y_train2, cmap='viridis', alpha=0.6)
axes[1].set_title('Split 2 (80/20)')
plt.tight_layout()
plt.savefig('cancer_scatter.png')
print("2.7 Scatter plots saved")

# 2.8 Box plot
df_plot = pd.DataFrame({'mean_area': X_scaled[:, 2], 'diagnosis': y})
plt.figure(figsize=(8, 6))
sns.boxplot(x='diagnosis', y='mean_area', data=df_plot)
plt.title('Mean Area Distribution')
plt.tight_layout()
plt.savefig('cancer_boxplot.png')
print("2.8 Box plot saved")

# 2.9 Train two models on both splits
lr1 = LogisticRegression(random_state=42, max_iter=1000)
dt1 = DecisionTreeClassifier(max_depth=5, random_state=42)
lr2 = LogisticRegression(random_state=42, max_iter=1000)
dt2 = DecisionTreeClassifier(max_depth=5, random_state=42)

lr1.fit(X_train1, y_train1)
dt1.fit(X_train1, y_train1)
lr2.fit(X_train2, y_train2)
dt2.fit(X_train2, y_train2)
print("2.9 4 models trained (LR & DT on both splits)")

# 2.10 Evaluate all models
def evaluate_classifier(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    return {
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred),
        'F1-Score': f1_score(y_test, y_pred),
        'ROC-AUC': roc_auc_score(y_test, y_proba)
    }

results = pd.DataFrame({
    'LR_Split1': evaluate_classifier(lr1, X_test1, y_test1),
    'DT_Split1': evaluate_classifier(dt1, X_test1, y_test1),
    'LR_Split2': evaluate_classifier(lr2, X_test2, y_test2),
    'DT_Split2': evaluate_classifier(dt2, X_test2, y_test2)
})
print(f"\n2.10 Model Comparison:\n{results}\n")


# =================================================================
# 3. HEART DISEASE (ADVANCED CLASSIFICATION)
# =================================================================

print("=" * 70)
print("3. HEART DISEASE - ADVANCED CLASSIFICATION")
print("=" * 70)

# Simulating heart disease data (replace with actual CSV)
np.random.seed(42)
n_samples = 303
heart_data = {
    'age': np.random.randint(29, 78, n_samples),
    'sex': np.random.randint(0, 2, n_samples),
    'cp': np.random.randint(0, 4, n_samples),
    'trestbps': np.random.randint(94, 200, n_samples),
    'chol': np.random.randint(126, 564, n_samples),
    'fbs': np.random.randint(0, 2, n_samples),
    'restecg': np.random.randint(0, 3, n_samples),
    'thalach': np.random.randint(71, 202, n_samples),
    'exang': np.random.randint(0, 2, n_samples),
    'oldpeak': np.random.uniform(0, 6.2, n_samples),
    'slope': np.random.randint(0, 3, n_samples),
    'ca': np.random.randint(0, 4, n_samples),
    'thal': np.random.randint(0, 4, n_samples),
    'target': np.random.randint(0, 2, n_samples)
}
df_heart = pd.DataFrame(heart_data)

# 3.1 Load dataset
print(f"\n3.1 Dataset shape: {df_heart.shape}")

# 3.2 Dataset info
num_features = df_heart.shape[1] - 1
num_classes = df_heart['target'].nunique()
print(f"3.2 Features: {num_features}, Classes: {num_classes}")
print(f"Class distribution:\n{df_heart['target'].value_counts()}")

# 3.3 Pairplot (first 4 features)
features_subset = df_heart.columns[:4].tolist() + ['target']
sns.pairplot(df_heart[features_subset], hue='target', diag_kind='kde', corner=True)
plt.savefig('heart_pairplot.png')
print("3.3 Pairplot saved")

# 3.4 Custom scatter plots
import itertools
features = df_heart.columns[:4].tolist()
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()
pairs = list(itertools.combinations(features, 2))
for idx, (f1, f2) in enumerate(pairs):
    if idx < len(axes):
        for target_val in df_heart['target'].unique():
            subset = df_heart[df_heart['target'] == target_val]
            axes[idx].scatter(subset[f1], subset[f2], label=f'Class {target_val}', alpha=0.6)
        axes[idx].set_xlabel(f1)
        axes[idx].set_ylabel(f2)
        axes[idx].legend()
plt.tight_layout()
plt.savefig('heart_scatter_custom.png')
print("3.4 Custom scatter plots saved")

# 3.5 Missing value handling
X_heart = df_heart.drop('target', axis=1)
y_heart = df_heart['target']
if X_heart.isnull().any().any():
    imputer = SimpleImputer(strategy='median')
    X_heart = pd.DataFrame(imputer.fit_transform(X_heart), columns=X_heart.columns)
print("3.5 Missing values handled")

# 3.6 Feature scaling
scaler = StandardScaler()
X_heart_scaled = scaler.fit_transform(X_heart)
X_heart_scaled = pd.DataFrame(X_heart_scaled, columns=X_heart.columns)
print("3.6 Features scaled")

# 3.7 Two stratified splits
X_train_h1, X_test_h1, y_train_h1, y_test_h1 = train_test_split(
    X_heart_scaled, y_heart, test_size=0.3, random_state=42, stratify=y_heart)
X_train_h2, X_test_h2, y_train_h2, y_test_h2 = train_test_split(
    X_heart_scaled, y_heart, test_size=0.2, random_state=42, stratify=y_heart)
print(f"3.7 Split1: {X_train_h1.shape}, Split2: {X_train_h2.shape}")

# 3.8 Train Logistic Regression on both splits
lr_h1 = LogisticRegression(random_state=42, max_iter=1000)
lr_h2 = LogisticRegression(random_state=42, max_iter=1000)
lr_h1.fit(X_train_h1, y_train_h1)
lr_h2.fit(X_train_h2, y_train_h2)
y_pred_h1 = lr_h1.predict(X_test_h1)
y_pred_h2 = lr_h2.predict(X_test_h2)
print("3.8 Logistic Regression trained on both splits")

# 3.9 Age vs Cholesterol plot
plt.figure(figsize=(10, 6))
for target_val in df_heart['target'].unique():
    subset = df_heart[df_heart['target'] == target_val]
    plt.scatter(subset['age'], subset['chol'], label=f'Target {target_val}', alpha=0.6)
plt.xlabel('Age')
plt.ylabel('Cholesterol')
plt.title('Age vs Cholesterol by Disease')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('heart_age_chol.png')
print("3.9 Age vs Cholesterol plot saved")

# 3.10 Performance metrics
metrics_h1 = evaluate_classifier(lr_h1, X_test_h1, y_test_h1)
metrics_h2 = evaluate_classifier(lr_h2, X_test_h2, y_test_h2)
print(f"\n3.10 Split 1 Metrics: {metrics_h1}")
print(f"Split 2 Metrics: {metrics_h2}")

# 3.11 Comparison table
comparison_heart = pd.DataFrame({'70/30': metrics_h1, '80/20': metrics_h2})
print(f"\n3.11 Comparison:\n{comparison_heart}")

# 3.12 Confusion matrices
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
cm1 = confusion_matrix(y_test_h1, y_pred_h1)
cm2 = confusion_matrix(y_test_h2, y_pred_h2)
sns.heatmap(cm1, annot=True, fmt='d', cmap='Blues', ax=axes[0])
axes[0].set_title('Confusion Matrix - Split 1')
sns.heatmap(cm2, annot=True, fmt='d', cmap='Blues', ax=axes[1])
axes[1].set_title('Confusion Matrix - Split 2')
plt.tight_layout()
plt.savefig('heart_confusion.png')
print("3.12 Confusion matrices saved\n")


# =================================================================
# 4. HEART DISEASE 2 (RANDOM FOREST vs SVM)
# =================================================================

print("=" * 70)
print("4. HEART DISEASE 2 - RANDOM FOREST vs SVM")
print("=" * 70)

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

# 4.1 Using same heart data, imputed
X_h2 = X_heart_scaled.copy()
y_h2 = y_heart.copy()
print(f"\n4.1 Dataset loaded")

# 4.2 Correlation analysis
temp_df = X_h2.copy()
temp_df['target'] = y_h2
correlations = temp_df.corr()['target'].abs().sort_values(ascending=False)
top5 = correlations[1:6]
print(f"4.2 Top 5 correlated features:\n{top5}")

# 4.3 Train-test split & scaling
X_train_h2, X_test_h2_new, y_train_h2_new, y_test_h2_new = train_test_split(
    X_h2, y_h2, test_size=0.3, random_state=42, stratify=y_h2)
print(f"4.3 Train: {X_train_h2.shape}, Test: {X_test_h2_new.shape}")

# 4.4 Train Random Forest
rf_model = RandomForestClassifier(n_estimators=120, max_depth=8, random_state=42)
rf_model.fit(X_train_h2, y_train_h2_new)
rf_pred = rf_model.predict(X_test_h2_new)
rf_proba = rf_model.predict_proba(X_test_h2_new)
print("4.4 Random Forest trained")

# 4.5 Train SVM
svm_model = SVC(kernel='rbf', probability=True, random_state=42)
svm_model.fit(X_train_h2, y_train_h2_new)
svm_pred = svm_model.predict(X_test_h2_new)
svm_proba = svm_model.predict_proba(X_test_h2_new)
print("4.5 SVM trained")

# 4.6 Evaluate both models
rf_metrics = evaluate_classifier(rf_model, X_test_h2_new, y_test_h2_new)
svm_metrics = evaluate_classifier(svm_model, X_test_h2_new, y_test_h2_new)
comparison_rf_svm = pd.DataFrame({'Random Forest': rf_metrics, 'SVM': svm_metrics})
print(f"\n4.6 Model Comparison:\n{comparison_rf_svm}")

# 4.7 Accuracy bar plot
models = ['Random Forest', 'SVM']
accuracies = [rf_metrics['Accuracy'], svm_metrics['Accuracy']]
plt.figure(figsize=(8, 6))
bars = plt.bar(models, accuracies, color=['steelblue', 'coral'], alpha=0.8)
for bar, acc in zip(bars, accuracies):
    plt.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
             f'{acc:.4f}', ha='center', va='bottom', fontweight='bold')
plt.ylabel('Accuracy')
plt.title('RF vs SVM Accuracy')
plt.ylim(0, 1)
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('heart2_accuracy.png')
print("4.7 Accuracy plot saved\n")


# =================================================================
# 5. TITANIC (NAIVE BAYES)
# =================================================================

print("=" * 70)
print("5. TITANIC - NAIVE BAYES CLASSIFICATION")
print("=" * 70)

from sklearn.preprocessing import MinMaxScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import roc_curve, auc

# Simulating Titanic data
np.random.seed(42)
n_passengers = 891
titanic_data = {
    'Pclass': np.random.choice([1, 2, 3], n_passengers),
    'Sex': np.random.choice(['male', 'female'], n_passengers),
    'Age': np.random.uniform(0.42, 80, n_passengers),
    'Fare': np.random.uniform(0, 512, n_passengers),
    'Embarked': np.random.choice(['C', 'Q', 'S'], n_passengers),
    'Survived': np.random.randint(0, 2, n_passengers)
}
df_titanic = pd.DataFrame(titanic_data)
df_titanic.loc[np.random.choice(df_titanic.index, 50), 'Age'] = np.nan

# 5.1 Load & select features
df_titanic = df_titanic.dropna(subset=['Survived'])
X_titanic = df_titanic.drop('Survived', axis=1)
y_titanic = df_titanic['Survived']
print(f"\n5.1 Dataset shape: {X_titanic.shape}")

# 5.2 Train-test split (80/20)
X_train_t, X_test_t, y_train_t, y_test_t = train_test_split(
    X_titanic, y_titanic, test_size=0.2, random_state=42, stratify=y_titanic)
print(f"5.2 Train: {X_train_t.shape}, Test: {X_test_t.shape}")

# 5.3 Separate numeric & categorical
numeric_features = ['Age', 'Fare']
categorical_features = ['Pclass', 'Sex', 'Embarked']
X_train_numeric = X_train_t[numeric_features]
X_test_numeric = X_test_t[numeric_features]
X_train_categorical = X_train_t[categorical_features]
X_test_categorical = X_test_t[categorical_features]
print("5.3 Features separated by type")

# 5.4 Impute numeric features
numeric_imputer = SimpleImputer(strategy='median')
X_train_numeric_imputed = numeric_imputer.fit_transform(X_train_numeric)
X_test_numeric_imputed = numeric_imputer.transform(X_test_numeric)
print("5.4 Numeric features imputed")

# 5.5 Scale numeric features
minmax_scaler = MinMaxScaler()
X_train_numeric_scaled = minmax_scaler.fit_transform(X_train_numeric_imputed)
X_test_numeric_scaled = minmax_scaler.transform(X_test_numeric_imputed)
print("5.5 Numeric features scaled (0-1)")

# 5.6 One-hot encode categorical
encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
X_train_categorical_encoded = encoder.fit_transform(X_train_categorical)
X_test_categorical_encoded = encoder.transform(X_test_categorical)
print(f"5.6 Categorical encoded: {X_train_categorical_encoded.shape[1]} features")

# 5.7 Combine features
X_train_final = np.hstack([X_train_categorical_encoded, X_train_numeric_scaled])
X_test_final = np.hstack([X_test_categorical_encoded, X_test_numeric_scaled])
print(f"5.7 Final feature matrix: {X_train_final.shape}")

# 5.8 Train Naive Bayes
nb_model = GaussianNB()
nb_model.fit(X_train_final, y_train_t)
y_pred_t = nb_model.predict(X_test_final)
y_proba_t = nb_model.predict_proba(X_test_final)[:, 1]
print("5.8 Naive Bayes trained")

# 5.9 Display predictions
comparison_t = pd.DataFrame({
    'Actual': y_test_t.values[:10],
    'Predicted': y_pred_t[:10],
    'Match': (y_test_t.values[:10] == y_pred_t[:10])
})
print(f"\n5.9 First 10 predictions:\n{comparison_t}")

# 5.10 Evaluate performance
accuracy_t = accuracy_score(y_test_t, y_pred_t)
precision_t = precision_score(y_test_t, y_pred_t, average='macro')
recall_t = recall_score(y_test_t, y_pred_t, average='macro')
f1_t = f1_score(y_test_t, y_pred_t, average='macro')
print(f"\n5.10 Accuracy: {accuracy_t:.4f}, Precision: {precision_t:.4f}")
print(f"Recall: {recall_t:.4f}, F1-Score: {f1_t:.4f}")

# 5.11 Visualizations (2x2 subplot)
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Plot 1: Survival count
survival_counts = y_test_t.value_counts()
axes[0, 0].bar(['Died', 'Survived'], survival_counts.values, color=['salmon', 'lightblue'])
axes[0, 0].set_title('Survival Distribution')
axes[0, 0].set_ylabel('Count')
axes[0, 0].grid(axis='y', alpha=0.3)

# Plot 2: Confusion matrix
cm_t = confusion_matrix(y_test_t, y_pred_t)
sns.heatmap(cm_t, annot=True, fmt='d', cmap='Blues', ax=axes[0, 1])
axes[0, 1].set_title('Confusion Matrix')
axes[0, 1].set_xlabel('Predicted')
axes[0, 1].set_ylabel('Actual')

# Plot 3: ROC curve
fpr, tpr, _ = roc_curve(y_test_t, y_proba_t)
roc_auc = auc(fpr, tpr)
axes[1, 0].plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC (AUC={roc_auc:.2f})')
axes[1, 0].plot([0, 1], [0, 1], 'navy', lw=2, linestyle='--', label='Random')
axes[1, 0].set_xlabel('False Positive Rate')
axes[1, 0].set_ylabel('True Positive Rate')
axes[1, 0].set_title('ROC Curve')
axes[1, 0].legend()
axes[1, 0].grid(alpha=0.3)

# Plot 4: Empty
axes[1, 1].axis('off')

plt.tight_layout()
plt.savefig('titanic_analysis.png')
print(f"5.11 Visualizations saved")
print(f"ROC-AUC Score: {roc_auc:.4f}\n")


# =================================================================
# 6. WINE (MULTI-CLASS CLASSIFICATION)
# =================================================================

print("=" * 70)
print("6. WINE - MULTI-CLASS CLASSIFICATION")
print("=" * 70)

from sklearn.datasets import load_wine

# 6.1 Load dataset
data_wine = load_wine()
X_wine = pd.DataFrame(data_wine.data, columns=data_wine.feature_names)
y_wine = pd.Series(data_wine.target, name='target')
print(f"\n6.1 Dataset shape: {X_wine.shape}, Classes: {y_wine.nunique()}")
print(f"Class distribution:\n{y_wine.value_counts()}")

# 6.2 Train-test split (80/20)
X_train_w, X_test_w, y_train_w, y_test_w = train_test_split(
    X_wine, y_wine, test_size=0.2, random_state=42, stratify=y_wine)
print(f"\n6.2 Train: {X_train_w.shape}, Test: {X_test_w.shape}")

# 6.3 Correlation heatmap
corr_matrix = X_train_w.corr()
plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0, square=True)
plt.title('Feature Correlation Heatmap (Training Data)')
plt.tight_layout()
plt.savefig('wine_correlation.png')
print("6.3 Correlation heatmap saved")

# 6.4 Feature scaling
scaler_w = StandardScaler()
X_train_w_scaled = scaler_w.fit_transform(X_train_w)
X_test_w_scaled = scaler_w.transform(X_test_w)
X_train_w_scaled = pd.DataFrame(X_train_w_scaled, columns=X_train_w.columns)
X_test_w_scaled = pd.DataFrame(X_test_w_scaled, columns=X_test_w.columns)
print("6.4 Features scaled")

# 6.5 Train Logistic Regression (multi-class OVR)
lr_wine = LogisticRegression(multi_class='ovr', random_state=42, max_iter=1000)
lr_wine.fit(X_train_w_scaled, y_train_w)
lr_pred_w = lr_wine.predict(X_test_w_scaled)
lr_proba_w = lr_wine.predict_proba(X_test_w_scaled)
print("6.5 Logistic Regression trained (OVR)")

# 6.6 Train Decision Tree
dt_wine = DecisionTreeClassifier(max_depth=4, random_state=42)
dt_wine.fit(X_train_w_scaled, y_train_w)
dt_pred_w = dt_wine.predict(X_test_w_scaled)
dt_proba_w = dt_wine.predict_proba(X_test_w_scaled)
print("6.6 Decision Tree trained")

# 6.7 Evaluate both models (multi-class)
def evaluate_multiclass(name, y_true, y_pred, y_proba):
    return {
        'Model': name,
        'Accuracy': accuracy_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred, average='weighted'),
        'Recall': recall_score(y_true, y_pred, average='weighted'),
        'F1-Score': f1_score(y_true, y_pred, average='weighted'),
        'AUC': roc_auc_score(y_true, y_proba, multi_class='ovr', average='weighted')
    }

lr_metrics_w = evaluate_multiclass('Logistic Regression', y_test_w, lr_pred_w, lr_proba_w)
dt_metrics_w = evaluate_multiclass('Decision Tree', y_test_w, dt_pred_w, dt_proba_w)
results_wine = pd.DataFrame([lr_metrics_w, dt_metrics_w]).set_index('Model')
print(f"\n6.7 Model Comparison:\n{results_wine}")

# 6.8 Accuracy comparison plot
models_w = ['Logistic Regression', 'Decision Tree']
accuracies_w = [lr_metrics_w['Accuracy'], dt_metrics_w['Accuracy']]
plt.figure(figsize=(10, 6))
bars = plt.bar(models_w, accuracies_w, color=['steelblue', 'coral'], alpha=0.8, edgecolor='black')
for bar, acc in zip(bars, accuracies_w):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
             f'{acc:.4f}', ha='center', va='bottom', fontsize=13, fontweight='bold')
plt.ylabel('Accuracy', fontsize=13, fontweight='bold')
plt.title('Wine Classification: Model Accuracy Comparison', fontsize=15, fontweight='bold')
plt.ylim(0, 1.1)
plt.grid(axis='y', alpha=0.3, linestyle='--')
plt.tight_layout()
plt.savefig('wine_accuracy.png')
print("6.8 Accuracy plot saved\n")

