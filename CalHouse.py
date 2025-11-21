import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

df = pd.read_csv("DATA/california_housing.csv")


train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)


imputer = SimpleImputer(strategy="median")


imputer.fit(train_df[["total_bedrooms"]])


train_df["total_bedrooms"] = imputer.transform(train_df[["total_bedrooms"]])
test_df["total_bedrooms"] = imputer.transform(test_df[["total_bedrooms"]])


from sklearn.preprocessing import OneHotEncoder


encoder = OneHotEncoder(handle_unknown='ignore')


encoder.fit(train_df[["ocean_proximity"]])


train_ocean = encoder.transform(train_df[["ocean_proximity"]]).toarray()
test_ocean = encoder.transform(test_df[["ocean_proximity"]]).toarray()


ocean_cols = encoder.get_feature_names_out(["ocean_proximity"])

train_ocean_df = pd.DataFrame(train_ocean, columns=ocean_cols, index=train_df.index)
test_ocean_df = pd.DataFrame(test_ocean, columns=ocean_cols, index=test_df.index)


train_df = train_df.drop("ocean_proximity", axis=1)
test_df = test_df.drop("ocean_proximity", axis=1)

train_df = pd.concat([train_df, train_ocean_df], axis=1)
test_df = pd.concat([test_df, test_ocean_df], axis=1)




from sklearn.preprocessing import MinMaxScaler

num_cols = [
    col for col in train_df.columns
    if train_df[col].dtype != "object" and not col.startswith("ocean_proximity_")
]

scaler = MinMaxScaler()

scaler.fit(train_df[num_cols])

train_df[num_cols] = scaler.transform(train_df[num_cols])
test_df[num_cols] = scaler.transform(test_df[num_cols])




from sklearn.tree import DecisionTreeRegressor
import pandas as pd


X_train = train_df.drop("median_house_value", axis=1)
y_train = train_df["median_house_value"]

X_test = test_df.drop("median_house_value", axis=1)
y_test = test_df["median_house_value"]


model = DecisionTreeRegressor(max_depth=6, random_state=42)
model.fit(X_train, y_train)


y_pred = model.predict(X_test)

results = pd.DataFrame({
    "Actual": y_test.iloc[:10].values,
    "Predicted": y_pred[:10]
})

print(results)

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("MAE :", mae)
print("RMSE:", rmse)
print("R²  :", r2)


import matplotlib.pyplot as plt

plt.figure(figsize=(6, 6))
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title("Actual vs Predicted (Decision Tree)")
plt.grid(True)
plt.show()


import pandas as pd
import matplotlib.pyplot as plt


importances = model.feature_importances_
feature_names = X_train.columns


feat_df = pd.DataFrame({
    "feature": feature_names,
    "importance": importances
})


top5 = feat_df.sort_values(by="importance", ascending=False).head(5)

plt.figure(figsize=(8, 4))
plt.barh(top5["feature"], top5["importance"])
plt.xlabel("Importance Score")
plt.title("Top 5 Most Important Features")
plt.gca().invert_yaxis()
plt.grid(True, axis='x')
plt.show()
