import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
import joblib

# Load the sensor data
df = pd.read_csv("data/simulated_sensors.csv")

# Drop the timestamp column for modeling
X = df[["temperature", "vibration", "current"]]

# Train Isolation Forest
model = IsolationForest(n_estimators=100, contamination=0.01, random_state=42)
model.fit(X)

# Predict anomalies (-1 = anomaly, 1 = normal)
df["anomaly"] = model.predict(X)

# Visualize temperature and mark anomalies
plt.figure(figsize=(14, 6))
plt.plot(df["timestamp"], df["temperature"], label="Temperature")
plt.scatter(df[df["anomaly"] == -1]["timestamp"], 
            df[df["anomaly"] == -1]["temperature"],
            color='red', label="Anomalies")
plt.title("Temperature with Anomalies")
plt.xlabel("Time")
plt.ylabel("Temperature (°C)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("data/temperature_anomalies.png")
plt.show()

# Save the model for the AI agent to use later
joblib.dump(model, "models/isolation_forest.pkl")

print("✅ Model trained, anomalies detected, and model saved.")
