import os
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_auc_score
)

# =========================================================
# VOLCANO ERUPTION WARNING SYSTEM
# =========================================================
# The supplied earthquake catalog does not contain an
# explicit eruption 0/1 label.
#
# Therefore, this project defines a clear forecasting target:
#   1 = eruption begins within the next 24 hours
#   0 = eruption does not begin within the next 24 hours
#
# The known Kilauea 2018 eruption onset is taken as
# 2018-05-03 00:00:00 for this project.
#
# IMPORTANT:
# This is a project-defined proxy label, not a label supplied
# directly in the earthquake file. Mention this limitation
# in the report and presentation.

FILE_PATH = "dataset/earthquake_data.txt"
ERUPTION_START = pd.Timestamp("2018-05-03 00:00:00")
WARNING_HOURS = 24

os.makedirs("results", exist_ok=True)

# =========================================================
# 1. LOAD DATA
# =========================================================
columns = [
    "year", "month", "day", "hour", "minute", "second",
    "latitude", "longitude", "depth", "magnitude", "event_id"
]

rows = []

with open(FILE_PATH, "r") as f:
    for line in f:
        if line.strip() and not line.lstrip().startswith("#"):
            parts = line.split()
            if len(parts) == 11:
                rows.append(parts)

df = pd.DataFrame(rows, columns=columns)

# Convert numerical columns
numeric_columns = [
    "year", "month", "day", "hour", "minute",
    "second", "latitude", "longitude", "depth", "magnitude"
]

for col in numeric_columns:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# Create timestamp
df["datetime"] = pd.to_datetime(
    df[["year", "month", "day", "hour", "minute", "second"]],
    errors="coerce"
)

df = df.dropna(subset=[
    "datetime", "latitude", "longitude", "depth", "magnitude"
])

df = df.sort_values("datetime").reset_index(drop=True)

# =========================================================
# 2. KEEP EARTHQUAKES BEFORE THE ERUPTION
# =========================================================
# Only earthquakes occurring before eruption onset can be
# used for a forward-looking warning prediction.

df = df[df["datetime"] < ERUPTION_START].copy()

# Time remaining until eruption
df["hours_to_eruption"] = (
    ERUPTION_START - df["datetime"]
).dt.total_seconds() / 3600

# Target:
# 1 = eruption begins within next 24 hours
# 0 = more than 24 hours away
df["warning"] = (
    df["hours_to_eruption"] <= WARNING_HOURS
).astype(int)

# =========================================================
# 3. FEATURES
# =========================================================
# We intentionally use the basic earthquake characteristics
# here to avoid making the target almost directly recoverable
# from time-dependent earthquake-rate features.

features = [
    "latitude",
    "longitude",
    "depth",
    "magnitude"
]

X = df[features].copy()
y = df["warning"].copy()

# =========================================================
# 4. TRAIN / TEST SPLIT
# =========================================================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y
)

# =========================================================
# 5. RANDOM FOREST
# =========================================================
model = RandomForestClassifier(
    n_estimators=300,
    max_depth=8,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

# =========================================================
# 6. PREDICTION
# =========================================================
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

accuracy = accuracy_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_prob)
cm = confusion_matrix(y_test, y_pred)

print("\n========== VOLCANO ERUPTION WARNING ==========")
print("Total earthquake records in original file: 44188")
print("Records before eruption onset:", len(df))
print("Warning window:", WARNING_HOURS, "hours")
print("\nTarget counts:")
print(y.value_counts())

print("\nFeatures:", features)
print("\nAccuracy:", round(accuracy, 4))
print("ROC-AUC:", round(auc, 4))

print("\nConfusion Matrix:")
print(cm)

print("\nClassification Report:")
print(classification_report(
    y_test,
    y_pred,
    target_names=["More than 24 hours away", "Within 24 hours"],
    zero_division=0
))

# =========================================================
# 7. SAVE RESULTS
# =========================================================
with open("results/model_results.txt", "w") as f:
    f.write("VOLCANO ERUPTION WARNING SYSTEM\n")
    f.write("=" * 45 + "\n")
    f.write(f"Original records: 44188\n")
    f.write(f"Pre-eruption records: {len(df)}\n")
    f.write(f"Warning window: {WARNING_HOURS} hours\n")
    f.write(f"Features: {features}\n\n")
    f.write("Target counts:\n")
    f.write(str(y.value_counts()) + "\n\n")
    f.write(f"Accuracy: {accuracy:.4f}\n")
    f.write(f"ROC-AUC: {auc:.4f}\n\n")
    f.write("Confusion Matrix:\n")
    f.write(str(cm) + "\n\n")
    f.write("Classification Report:\n")
    f.write(classification_report(
        y_test,
        y_pred,
        target_names=["More than 24 hours away", "Within 24 hours"],
        zero_division=0
    ))
    f.write("\n\nIMPORTANT LIMITATION:\n")
    f.write(
        "The earthquake catalog does not provide an explicit eruption label. "
        "The eruption onset and 24-hour warning target are project-defined "
        "using the known 2018 Kilauea eruption sequence.\n"
    )

# =========================================================
# 8. CONFUSION MATRIX GRAPH
# =========================================================
disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["More than 24h", "Within 24h"]
)

disp.plot()
plt.title("Random Forest Confusion Matrix")
plt.tight_layout()
plt.savefig("results/confusion_matrix.png", dpi=200)
plt.close()

# =========================================================
# 9. FEATURE IMPORTANCE GRAPH
# =========================================================
importance = pd.Series(
    model.feature_importances_,
    index=features
).sort_values()

plt.figure(figsize=(8, 5))
plt.barh(importance.index, importance.values)
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.title("Random Forest Feature Importance")
plt.tight_layout()
plt.savefig("results/feature_importance.png", dpi=200)
plt.close()

print("\nGraphs and results saved in the results/ folder.")
