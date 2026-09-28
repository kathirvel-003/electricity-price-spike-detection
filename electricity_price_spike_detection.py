# Electricity Market Price Spike Detection
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix

df = pd.read_csv("electricity_market_price_spike_dataset.csv")

print(df.head())
print(df.info())
print(df.describe())
print("\nMissing values:\n", df.isnull().sum())

# Price trend
plt.figure(figsize=(12,5))
plt.plot(df["Electricity_Price"])
plt.title("Electricity Market Price Trend")
plt.xlabel("Observation")
plt.ylabel("Electricity Price")
plt.show()

print("\nSpike counts:\n", df["Price_Spike"].value_counts())

features = ["Hour","Demand_MW","Renewable_Generation_MW","Temperature_C",
            "Wind_Speed_kmh","Day_of_Week","Is_Weekend"]
X, y = df[features], df["Price_Spike"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# Baseline
log_model = LogisticRegression(max_iter=1000)
log_model.fit(X_train, y_train)
log_pred = log_model.predict(X_test)

print("\n--- Logistic Regression ---")
print("Accuracy :", accuracy_score(y_test, log_pred))
print("Precision:", precision_score(y_test, log_pred, zero_division=0))
print("Recall   :", recall_score(y_test, log_pred, zero_division=0))
print("F1 Score :", f1_score(y_test, log_pred, zero_division=0))

# Main model
rf_model = RandomForestClassifier(
    n_estimators=200, random_state=42, class_weight="balanced"
)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)

print("\n--- Random Forest ---")
print("Accuracy :", accuracy_score(y_test, rf_pred))
print("Precision:", precision_score(y_test, rf_pred, zero_division=0))
print("Recall   :", recall_score(y_test, rf_pred, zero_division=0))
print("F1 Score :", f1_score(y_test, rf_pred, zero_division=0))
print(classification_report(y_test, rf_pred, zero_division=0))

# Confusion matrix
cm = confusion_matrix(y_test, rf_pred)
plt.figure(figsize=(5,4))
plt.imshow(cm)
plt.title("Random Forest Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.colorbar()
plt.show()

# Feature importance
importance = pd.Series(rf_model.feature_importances_, index=features).sort_values(ascending=False)
print("\nFeature Importance:\n", importance)

plt.figure(figsize=(8,5))
importance.sort_values().plot(kind="barh")
plt.title("Random Forest Feature Importance")
plt.xlabel("Importance")
plt.show()

# New prediction example
new_data = pd.DataFrame({
    "Hour":[18], "Demand_MW":[4700], "Renewable_Generation_MW":[300],
    "Temperature_C":[35], "Wind_Speed_kmh":[8],
    "Day_of_Week":[2], "Is_Weekend":[0]
})
prediction = rf_model.predict(new_data)[0]
probability = rf_model.predict_proba(new_data)[0][1]
print("\nPrediction:", "PRICE SPIKE DETECTED" if prediction == 1 else "NORMAL PRICE")
print("Spike probability:", round(probability*100,2), "%")
