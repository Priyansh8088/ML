# CalHouse

# FLOW SUMMARY

1. Load data
2. Split train/test
3. Median-impute missing values
4. One-hot encode the category
5. Scale numeric features
6. Split X/y
7. Train Decision Tree
8. Predict on test
9. Show sample predictions
10. Compute MAE, RMSE, R²
11. Scatter plot actual vs predicted
12. Plot top 5 most important features

---


# Cancer
# **🔹 Short Flow Explanation**

### **1. Load dataset**

You load the breast cancer dataset into `X` (features) and `y` (target).

---

### **2. Dataset overview**

You print shape, feature names, and class distribution.
You compute correlations, pick the top 5 features most linked to the target, and plot a heatmap.

---

### **3. Add missing values + preprocess**

You randomly insert ~3% NaNs.
You fill them using median imputation.
You standardize all features with `StandardScaler`.

---

### **4. Two data splits**

You create:

* 70/30 train-test split
* 80/20 train-test split

This is used to check performance consistency.

---

### **5. Visualizations**

You plot:

* Scatter plots: *mean radius vs mean texture* for both splits
* Box plots: *mean area* distribution for benign vs malignant

Shows class separability.

---

### **6. Train models**

You train two classifiers on each split:

* Logistic Regression
* Decision Tree

---

### **7. Evaluate**

You compute:

* Accuracy
* Precision
* Recall
* F1-score
* ROC-AUC

Then print a table comparing both models across both splits.

---

# Heart


### **1. Load dataset**

You read the heart-disease CSV into a DataFrame.

---

### **2. Basic dataset info**

You calculate:

* number of features
* number of classes
* class distribution

---

### **3. Pairplots (Seaborn + Matplotlib)**

Using the first 4 features, you generate:

* a Seaborn pairplot
* custom Matplotlib scatter plots

These show relationships between features and how classes separate visually.

---

### **4. Missing value check & imputation**

You check for NaNs.
If found, you replace them with **median of each feature**.

---

### **5. Feature scaling**

All input features are standardized using **StandardScaler**
(mean = 0, std = 1).

---

### **6. Train–test splits (two setups)**

You create:

* **70–30 split**
* **80–20 split**

Both use stratification to preserve class balance.

---

### **7. Logistic Regression (on both splits)**

You train a Logistic Regression model on:

* split 1 (70–30)
* split 2 (80–20)

You predict labels for each test set.

---

### **8. Age vs Cholesterol scatter plot**

A separate plot that shows how age and cholesterol values differ across disease classes.

---

### **9. Performance metrics**

For each split, you compute:

* Accuracy
* Precision
* Recall
* F1 score
* ROC-AUC

You print metrics for both splits.

---

### **10. Comparison table**

You create a **side-by-side DataFrame** comparing 70–30 vs 80–20 performance.

---

### **11. Confusion matrices**

You plot confusion matrices for:

* 70–30
* 80–20

This shows how well the model captured true positives/negatives.

---

### **12. Save images**

Pairplots + confusion matrices are exported as PNG files.

---

# Heart2


### **1. Load dataset + handle missing values**

* Read the heart disease dataset.
* Replace all missing numeric values with the column median.
* Split into:

  * **X** → features
  * **y** → target

---

### **2. Correlation analysis**

* Compute correlations with the `target` column.
* Print the **top 5 most strongly correlated features** (important predictors).

---

### **3. Train–test split + scaling**

* Split into **70% train / 30% test**.
* Apply **StandardScaler**:

  * Fit on training data
  * Transform both train & test
    (Prepares data for ML models)

---

### **4. Train two models**

* **Random Forest** (120 trees, depth=8)
* **SVM (RBF kernel)**
  Both models are trained on the scaled training data.

---

### **5. Evaluate both models**

For each model, compute on the test set:

* Accuracy
* Precision
* Recall
* F1-Score
* ROC-AUC

Store everything in a comparison table and print it.

---

### **6. Accuracy bar plot**

* Plot a bar chart comparing the **accuracy** of Random Forest vs SVM.
* Add value labels above each bar.

---

# titanic
Here’s the **clean, short explanation** of your entire Titanic Naive Bayes pipeline — simple, clear, no fluff.


### **1. Load dataset**

* Load the Titanic CSV.
* Keep only: `Pclass`, `Sex`, `Age`, `Fare`, `Embarked`, `Survived`.
* Drop rows with missing target values.
* Split features (X) and target (y).

---

### **2. Train–test split (80/20)**

You create separate training and testing sets.

---

### **3. Data preparation**

You process numerical + categorical data separately:

#### **• Numerical features (`Age`, `Fare`)**

* Impute missing values using median (fit on train, apply to test).
* Scale with MinMaxScaler (fit on train, apply to test).

#### **• Categorical features (`Pclass`, `Sex`, `Embarked`)**

* One-hot encode using OneHotEncoder.
* Train encoder on training set only.
* Transform both train and test.

#### **• Combine processed features**

Concatenate encoded categorical + scaled numerical features into final training and testing matrices.

---

### **4. Train Naive Bayes classifier**

* Fit **GaussianNB** on the processed training features.
* Predict labels and probability scores for the test set.

---

### **5. Display predictions**

Show the first 10 rows of:

* Actual survival
* Model prediction

Good for quick sanity check.

---

### **6. Evaluate performance**

Compute and print:

* Accuracy
* Precision (macro)
* Recall (macro)
* F1-score (macro)

---

### **7. Visualizations (2×2 subplot)**

#### **Plot 1 — Survival count**

Shows how many passengers survived vs not in the test set.

#### **Plot 2 — Confusion matrix**

Visual heatmap of:

* TP, FP
* TN, FN

Shows how the classifier is performing.

#### **Plot 3 — ROC curve**

Plots:

* True Positive Rate
* False Positive Rate
  Also shows the **ROC-AUC score**.

#### **Plot 4 — Empty**

Unused subplot turned off.

---

### **8. Print final AUC score**

Shows the ROC-AUC value separately at the end.

---

# Wine


### **1. Load dataset**

You read the wine dataset and split it into:

* **X** → all chemical measurements
* **y** → wine class (target)

Then you create an **80-20 train–test split**.

---

### **2. Print train/test sizes**

Just shows how many samples go into training vs testing.

---

### **3. Correlation heatmap**

You plot a heatmap of correlations among all **training features** to see:

* which chemical attributes move together
* which might be redundant
* which might be strong predictors

---

### **4. Feature scaling**

You apply **StandardScaler**:

* fit on training features
* transform both train & test
  Converted back into DataFrames to preserve column names.

(Standardization is critical because Logistic Regression and Decision Trees behave differently with unscaled data.)

---

### **5. Train two models**

You train:

* **Logistic Regression** (one-vs-rest for multiclass)
* **Decision Tree** (depth=4)

Both models are fitted on the **scaled training data**.

---

### **6. Evaluate both models**

For each model, you compute:

* Accuracy
* Precision (weighted)
* Recall (weighted)
* F1-Score (weighted)
* AUC (multiclass OVR)

Then you store everything in a table and print it.

---

### **7. Accuracy comparison plot**

You extract the accuracy of both models and plot a bar chart showing:

* Logistic Regression accuracy
* Decision Tree accuracy

Labels above the bars display the exact values.

---






