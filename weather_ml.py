"""
Weather Comfort ML Pipeline
============================
Models: Random Forest · Gradient Boosting · Logistic Regression
Clustering: K-Means (4 clusters)
Target: predict whether an hour is comfortable for outdoor activity / surveillance
"""

import pandas as pd
import numpy as np
import json
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, accuracy_score
)

# ── 1. Load data ───────────────────────────────────────────────────────────────
df = pd.read_csv("weather_data.csv")

df["hour"]  = [i % 24 for i in range(len(df))]
df["month"] = [(i // 24 // 30) % 12 + 1 for i in range(len(df))]

T       = df["Outdoor Drybulb Temperature [C]"]
RH      = df["Outdoor Relative Humidity [%]"]
direct  = df["Direct Solar Radiation [W/m2]"]
diffuse = df["Diffuse Solar Radiation [W/m2]"]
solar   = direct + diffuse

# ── 2. Feature engineering ─────────────────────────────────────────────────────
df["heat_index"] = (
    -8.78 + 1.61*T + 2.34*RH
    - 0.146*T*RH - 0.012*T**2 - 0.016*RH**2
    + 0.0022*T**2*RH + 0.000725*T*RH**2
)
df["total_solar"]         = solar
df["hour_sin"]            = np.sin(2 * np.pi * df["hour"] / 24)
df["hour_cos"]            = np.cos(2 * np.pi * df["hour"] / 24)
df["month_sin"]           = np.sin(2 * np.pi * df["month"] / 12)
df["month_cos"]           = np.cos(2 * np.pi * df["month"] / 12)
df["temp_rh_interaction"] = T * RH
df["is_daytime"]          = ((df["hour"] >= 6) & (df["hour"] <= 18)).astype(int)

# ── 3. Target variable ─────────────────────────────────────────────────────────
# Comfortable = temp 18-28°C  AND  humidity 30-60%  AND  solar < 200 W/m²
df["is_comfortable"] = (
    (T >= 18) & (T <= 28) &
    (RH >= 30) & (RH <= 60) &
    (solar < 200)
).astype(int)

print(f"Class balance — Comfortable: {df['is_comfortable'].sum()} "
      f"| Uncomfortable: {(df['is_comfortable']==0).sum()}")

# ── 4. Feature list ────────────────────────────────────────────────────────────
FEATURES = [
    "Outdoor Drybulb Temperature [C]",
    "Outdoor Relative Humidity [%]",
    "Direct Solar Radiation [W/m2]",
    "Diffuse Solar Radiation [W/m2]",
    "6h Prediction Outdoor Drybulb Temperature [C]",
    "12h Prediction Outdoor Drybulb Temperature [C]",
    "24h Prediction Outdoor Drybulb Temperature [C]",
    "6h Prediction Outdoor Relative Humidity [%]",
    "12h Prediction Outdoor Relative Humidity [%]",
    "24h Prediction Outdoor Relative Humidity [%]",
    "6h Prediction Direct Solar Radiation [W/m2]",
    "12h Prediction Direct Solar Radiation [W/m2]",
    "24h Prediction Direct Solar Radiation [W/m2]",
    "heat_index", "total_solar",
    "hour_sin", "hour_cos",
    "month_sin", "month_cos",
    "temp_rh_interaction", "is_daytime",
]

X = df[FEATURES]
y = df["is_comfortable"]

# ── 5. Train / test split ──────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

# ── 6. Train models ────────────────────────────────────────────────────────────
print("\nTraining models...")

rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
gb = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42)
lr = LogisticRegression(max_iter=1000, random_state=42)

rf.fit(X_train, y_train)
gb.fit(X_train, y_train)
lr.fit(X_train_s, y_train)

# ── 7. Evaluate ────────────────────────────────────────────────────────────────
model_map = {
    "Random Forest":       (rf, X_test),
    "Gradient Boosting":   (gb, X_test),
    "Logistic Regression": (lr, X_test_s),
}

results = {}
print("\n" + "="*60)
for name, (model, Xt) in model_map.items():
    pred = model.predict(Xt)
    prob = model.predict_proba(Xt)[:, 1]
    acc  = round(accuracy_score(y_test, pred) * 100, 2)
    auc  = round(roc_auc_score(y_test, prob) * 100, 2)
    cm   = confusion_matrix(y_test, pred).tolist()
    results[name] = {"accuracy": acc, "auc": auc, "cm": cm}
    print(f"\n{name}  —  Accuracy: {acc}%  |  AUC: {auc}%")
    print(classification_report(y_test, pred, target_names=["Uncomfortable", "Comfortable"]))
    print(f"Confusion matrix: {cm}")

# ── 8. Feature importances ─────────────────────────────────────────────────────
importances = (
    pd.Series(rf.feature_importances_, index=FEATURES)
    .sort_values(ascending=False)
    .head(10)
    .round(4)
)
print("\n=== Top 10 Feature Importances (Random Forest) ===")
print(importances)

# ── 9. 5-fold cross-validation ─────────────────────────────────────────────────
cv_scores = cross_val_score(rf, X, y, cv=5, scoring="accuracy")
print(f"\n5-Fold CV  —  Mean: {cv_scores.mean()*100:.2f}%  |  Std: {cv_scores.std()*100:.2f}%")

# ── 10. K-Means clustering ─────────────────────────────────────────────────────
print("\nRunning K-Means clustering (k=4)...")

CLUSTER_FEATS = [
    "Outdoor Drybulb Temperature [C]",
    "Outdoor Relative Humidity [%]",
    "total_solar", "heat_index", "hour", "month",
]
Xc_scaled = StandardScaler().fit_transform(df[CLUSTER_FEATS])
km = KMeans(n_clusters=4, random_state=42, n_init=10)
df["cluster"] = km.fit_predict(Xc_scaled)

cluster_summary = (
    df.groupby("cluster")
    .agg(
        avg_temp=("Outdoor Drybulb Temperature [C]", "mean"),
        avg_rh=("Outdoor Relative Humidity [%]", "mean"),
        avg_solar=("total_solar", "mean"),
        comfort_pct=("is_comfortable", "mean"),
        count=("is_comfortable", "count"),
    )
    .round(2)
)
cluster_summary["comfort_pct"] = (cluster_summary["comfort_pct"] * 100).round(1)
print("\nCluster Profiles:")
print(cluster_summary.to_string())

# Assign human-readable labels
LABELS = {
    cluster_summary["avg_temp"].idxmax():   "Hot & Humid",
    cluster_summary["avg_solar"].idxmax():  "Peak Solar",
    cluster_summary["avg_rh"].idxmax():     "Cool & Foggy",
    cluster_summary["comfort_pct"].idxmax(): "Best Conditions",
}
# fallback for any unassigned cluster
for c in range(4):
    LABELS.setdefault(c, f"Cluster {c}")

cluster_summary["label"] = [LABELS.get(i, f"Cluster {i}") for i in cluster_summary.index]
print("\nCluster labels:", cluster_summary["label"].to_dict())

# ── 11. Predict on full dataset ────────────────────────────────────────────────
df["rf_pred"]   = rf.predict(X)
df["rf_prob"]   = rf.predict_proba(X)[:, 1]
df["gb_pred"]   = gb.predict(X)
df["gb_prob"]   = gb.predict_proba(X)[:, 1]

# Best hours by predicted probability
hourly_pred = (
    df.groupby("hour")
    .agg(
        avg_rf_prob=("rf_prob", "mean"),
        avg_gb_prob=("gb_prob", "mean"),
        avg_temp=("Outdoor Drybulb Temperature [C]", "mean"),
        avg_rh=("Outdoor Relative Humidity [%]", "mean"),
        avg_solar=("total_solar", "mean"),
    )
    .round(3)
)
print("\n=== Hourly Predicted Comfort Probability (RF) ===")
print(hourly_pred[["avg_rf_prob", "avg_temp", "avg_rh", "avg_solar"]].to_string())

# ── 12. Save full results to CSV ───────────────────────────────────────────────
df.to_csv("ml_full_predictions.csv", index=False)
cluster_summary.to_csv("cluster_profiles.csv")
importances.to_frame("importance").to_csv("feature_importances.csv")
hourly_pred.to_csv("hourly_predictions.csv")

print("\nSaved: ml_full_predictions.csv, cluster_profiles.csv,")
print("       feature_importances.csv, hourly_predictions.csv")
