# train_model.py
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
import joblib
import os

os.makedirs("models", exist_ok=True)

# Define prototypes for items: mean RGB (0-255) and mean size (cm)
prototypes = {
    "Apple":      {"rgb": (200, 30, 30),  "size": 7.0},   # red, ~7cm
    "Banana":     {"rgb": (255, 225, 53), "size": 18.0},  # yellow, ~18cm
    "Orange":     {"rgb": (255, 140, 0),  "size": 8.5},   # orange, ~8.5cm
    "Lime":       {"rgb": (50, 205, 50),  "size": 6.0},   # green, small
    "Blueberry":  {"rgb": (60, 50, 150),  "size": 1.0},   # bluish, tiny
    "Tomato":     {"rgb": (230, 70, 60),  "size": 6.5},   # tomato-like red
    "Eggplant":   {"rgb": (120, 40, 150), "size": 12.0},  # purple, long-ish
    "Potato":     {"rgb": (200, 170, 120),"size": 6.5},   # brownish
}

np.random.seed(42)

rows = []
samples_per_class = 220
for label, proto in prototypes.items():
    r_mean, g_mean, b_mean = proto["rgb"]
    size_mean = proto["size"]
    for _ in range(samples_per_class):
        # Add some gaussian noise to simulate real-world variation
        r = np.clip(np.random.normal(r_mean, 18), 0, 255)
        g = np.clip(np.random.normal(g_mean, 18), 0, 255)
        b = np.clip(np.random.normal(b_mean, 18), 0, 255)
        size = np.clip(np.random.normal(size_mean, size_mean*0.12), 0.2, 50)
        rows.append([r, g, b, size, label])

df = pd.DataFrame(rows, columns=["r", "g", "b", "size_cm", "label"])

# Shuffle
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Features & labels
X = df[["r", "g", "b", "size_cm"]].values
y = df["label"].values

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train KNN
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_scaled, y)

# Save
joblib.dump(knn, "models/knn_model.pkl")
joblib.dump(scaler, "models/scaler.pkl")
df.to_csv("models/synthetic_dataset.csv", index=False)

print("Training finished. Models saved to /models.")
print("Sample dataset saved to models/synthetic_dataset.csv")
print("Class distribution:")
print(df['label'].value_counts())
