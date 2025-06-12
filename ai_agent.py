import pandas as pd
import numpy as np
import joblib
import time
from datetime import datetime
import paho.mqtt.client as mqtt


# Load the trained Isolation Forest model
model = joblib.load("models/isolation_forest.pkl")

# Simulate live sensor readings
def generate_live_data():
    temperature = np.random.normal(loc=60, scale=2)
    vibration = np.random.normal(loc=0.5, scale=0.1)
    current = np.random.normal(loc=5, scale=0.5)

    # 5% chance of injecting a fault
    if np.random.rand() < 0.05:
        temperature += np.random.uniform(10, 20)  # Overheat
        vibration += np.random.uniform(0.5, 1.0)  # Extra shake
        current += np.random.uniform(1.5, 3.0)    # Power spike

    return {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "temperature": round(temperature, 2),
        "vibration": round(vibration, 2),
        "current": round(current, 2)
    }

# Run the agent
print("🚀 AI Agent started. Monitoring incoming sensor data...\n")

# MQTT Setup
MQTT_BROKER = "broker.hivemq.com"  # Public broker for demo
MQTT_PORT = 1883
MQTT_TOPIC = "smartfactory/alerts"

client = mqtt.Client()
client.connect(MQTT_BROKER, MQTT_PORT, 60)
client.loop_start()


for _ in range(50):  # Simulate 50 readings (can be infinite)
    reading = generate_live_data()
    X = [[reading["temperature"], reading["vibration"], reading["current"]]]
    prediction = model.predict(X)[0]

    if prediction == -1:
        status = "❌ ANOMALY DETECTED"
    else:
        status = "✅ Normal"

    # print(f"[{reading['timestamp']}] Temp: {reading['temperature']}°C, "
    #       f"Vib: {reading['vibration']}g, Curr: {reading['current']}A → {status}")
    msg = f"[{reading['timestamp']}] Temp: {reading['temperature']}°C, Vib: {reading['vibration']}g, Curr: {reading['current']}A"

    if prediction == -1:
        status = "❌ ANOMALY DETECTED"
        alert_msg = f"⚠️ ALERT: {msg}"
        client.publish(MQTT_TOPIC, alert_msg)
    else:
        status = "✅ Normal"

    print(f"{msg} → {status}")

    time.sleep(1)  # Wait 1 second to simulate real-time streaming
