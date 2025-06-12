import pandas as pd
import numpy as np

# Set seed for repeatability
np.random.seed(42)

# Generate 1000 time-stamped entries (once per minute)
rows = 1000
timestamps = pd.date_range(start='2025-01-01', periods=rows, freq='T')

# Simulate normal values (mean ± std deviation)
temperature = np.random.normal(loc=60, scale=2, size=rows)  # degrees Celsius
vibration = np.random.normal(loc=0.5, scale=0.1, size=rows)  # in g
current = np.random.normal(loc=5, scale=0.5, size=rows)  # in Amperes

# Inject some anomalies to mimic machine issues
temperature[300:310] += 15   # Overheating
vibration[700:710] += 1.2    # Bearing failure (increased vibration)
current[500:505] += 2.0      # Power surge

# Combine into DataFrame
df = pd.DataFrame({
    "timestamp": timestamps,
    "temperature": temperature,
    "vibration": vibration,
    "current": current
})

# Save to CSV for later use
df.to_csv("data/simulated_sensors.csv", index=False)

print("✅ Sensor data generated and saved to 'data/simulated_sensors.csv'")
