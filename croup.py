# ============================================================
# CROP RECOMMENDATION SYSTEM USING SOIL DATA
# Domain: Data Science / Agriculture
# Platform: Google Colab
# Algorithms: Decision Tree and Random Forest
# ============================================================

# 1. IMPORT LIBRARIES
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from google.colab import files
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

print("=" * 60)
print("       CROP RECOMMENDATION SYSTEM")
print("=" * 60)

# 2. UPLOAD DATASET
print("\nUpload Crop_recommendation.csv")
uploaded = files.upload()

filename = next(iter(uploaded))
df = pd.read_csv(filename)

print("\nDataset loaded successfully!")
print("Dataset shape:", df.shape)
print("\nFirst five records:")
print(df.head())

# 3. DATA PREPROCESSING
print("\nMissing values before cleaning:")
print(df.isnull().sum())

df = df.drop_duplicates()
df = df.dropna().reset_index(drop=True)

features = [
    "N", "P", "K", "temperature",
    "humidity", "ph", "rainfall"
]

target = "label"

# Check required columns
required_columns = features + [target]
missing_columns = [
    col for col in required_columns if col not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Dataset is missing required columns: {missing_columns}"
    )

# Ensure input features contain valid numeric values
for col in features:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.dropna(subset=required_columns).reset_index(drop=True)

if df[target].nunique() < 2:
    raise ValueError("At least two crop classes are required.")

print("\nCleaned dataset shape:", df.shape)
print("Number of crop types:", df[target].nunique())

# 4. EXPLORATORY DATA ANALYSIS
plt.figure(figsize=(13, 5))
df[target].value_counts().plot(kind="bar", color="green")
plt.title("Distribution of Crop Types")
plt.xlabel("Crop")
plt.ylabel("Number of Samples")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

plt.figure(figsize=(9, 6))
sns.heatmap(
    df[features].corr(),
    annot=True,
    cmap="YlGnBu",
    fmt=".2f"
)
plt.title("Correlation Between Soil and Weather Features")
plt.tight_layout()
plt.show()

# 5. FEATURE SELECTION AND DATA SPLITTING
X = df[features]
y = df[target]

# Stratification requires at least two samples per class
class_counts = y.value_counts()
if class_counts.min() < 2:
    raise ValueError(
        "Each crop class must have at least two samples "
        "to use a stratified train-test split."
    )

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))

# 6. TRAIN DECISION TREE MODEL
dt_model = DecisionTreeClassifier(
    max_depth=12,
    random_state=42
)

dt_model.fit(X_train, y_train)
dt_predictions = dt_model.predict(X_test)

dt_accuracy = accuracy_score(y_test, dt_predictions)

# 7. TRAIN RANDOM FOREST MODEL
rf_model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    n_jobs=-1
)

rf_model.fit(X_train, y_train)
rf_predictions = rf_model.predict(X_test)

rf_accuracy = accuracy_score(y_test, rf_predictions)

# 8. COMPARE MODEL ACCURACY
print("\n" + "=" * 60)
print("             MODEL EVALUATION")
print("=" * 60)

results = pd.DataFrame({
    "Algorithm": ["Decision Tree", "Random Forest"],
    "Accuracy (%)": [
        dt_accuracy * 100,
        rf_accuracy * 100
    ]
})

print(results.round(2).to_string(index=False))

plt.figure(figsize=(7, 5))
plt.bar(
    results["Algorithm"],
    results["Accuracy (%)"],
    color=["orange", "forestgreen"]
)
plt.title("Model Accuracy Comparison")
plt.ylabel("Accuracy (%)")
plt.ylim(0, 100)

for i, value in enumerate(results["Accuracy (%)"]):
    plt.text(i, value + 1, f"{value:.2f}%", ha="center")

plt.tight_layout()
plt.show()

# 9. CLASSIFICATION REPORT
print("\nRANDOM FOREST CLASSIFICATION REPORT")
print(classification_report(y_test, rf_predictions))

# 10. CONFUSION MATRIX
cm = confusion_matrix(
    y_test,
    rf_predictions,
    labels=rf_model.classes_
)

plt.figure(figsize=(13, 10))
sns.heatmap(
    cm,
    cmap="Greens",
    xticklabels=rf_model.classes_,
    yticklabels=rf_model.classes_
)
plt.title("Random Forest Confusion Matrix")
plt.xlabel("Predicted Crop")
plt.ylabel("Actual Crop")
plt.xticks(rotation=90)
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()

# 11. FEATURE IMPORTANCE
importance = pd.DataFrame({
    "Feature": features,
    "Importance": rf_model.feature_importances_
}).sort_values("Importance", ascending=False)

print("\nFEATURE IMPORTANCE")
print(importance.round(4).to_string(index=False))

plt.figure(figsize=(8, 5))
plt.barh(
    importance["Feature"],
    importance["Importance"],
    color="seagreen"
)
plt.gca().invert_yaxis()
plt.title("Feature Importance - Random Forest")
plt.xlabel("Importance")
plt.tight_layout()
plt.show()

# 12. USER INPUT AND CROP PREDICTION
print("\n" + "=" * 60)
print("          ENTER SOIL AND WEATHER DATA")
print("=" * 60)

try:
    N = float(input("Nitrogen (N): "))
    P = float(input("Phosphorus (P): "))
    K = float(input("Potassium (K): "))
    temperature = float(input("Temperature (Celsius): "))
    humidity = float(input("Humidity (%): "))
    ph = float(input("Soil pH: "))
    rainfall = float(input("Rainfall (mm): "))

    values = [N, P, K, temperature, humidity, ph, rainfall]

    if not all(np.isfinite(v) for v in values):
        raise ValueError("Please enter finite numeric values.")

    new_data = pd.DataFrame([values], columns=features)

    # Predict with both algorithms
    dt_crop = dt_model.predict(new_data)[0]
    rf_crop = rf_model.predict(new_data)[0]

    print("\n" + "=" * 60)
    print("          CROP RECOMMENDATION RESULT")
    print("=" * 60)

    print("Decision Tree Recommendation:", dt_crop)
    print("Random Forest Recommendation:", rf_crop)

    # Choose the model with higher held-out test accuracy
    if rf_accuracy >= dt_accuracy:
        best_model = rf_model
        best_name = "Random Forest"
    else:
        best_model = dt_model
        best_name = "Decision Tree"

    recommended_crop = best_model.predict(new_data)[0]

    print("\nSelected Model:", best_name)
    print("RECOMMENDED CROP:", recommended_crop)

    # Show the model's relative class probabilities
    probabilities = best_model.predict_proba(new_data)[0]
    classes = best_model.classes_
    top_indices = np.argsort(probabilities)[::-1][:3]

    print("\nTop predicted crop classes:")
    for i in top_indices:
        print(f"{classes[i]}: {probabilities[i] * 100:.2f}%")

    print("\nRecommendation generated successfully!")

except ValueError as error:
    print("\nInvalid input:", error)

print("\nNote: Predictions are based on the training dataset.")
print("Validate recommendations with local agricultural experts.")
print("=" * 60)